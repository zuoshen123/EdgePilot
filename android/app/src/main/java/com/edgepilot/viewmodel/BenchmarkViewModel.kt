package com.edgepilot.viewmodel

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.edgepilot.harness.SamplerWindow
import com.edgepilot.native.NativeEngine
import java.util.Locale
import org.json.JSONObject
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class BenchmarkResult(
    val ttftMs: Float = 0f,
    val itlAvgMs: Float = 0f,
    val itlP50Ms: Float = 0f,
    val itlP90Ms: Float = 0f,
    val itlP99Ms: Float = 0f,
    val tokensPerSec: Float = 0f,
    val totalTokens: Int = 0,
    val totalTimeMs: Float = 0f,
    val itlSeries: List<Double> = emptyList(),
    val powerTimeline: List<Float> = emptyList(),   // 单跑窗口功耗曲线（mW，10Hz；空=无数据）
    val energyMj: Float = 0f,                       // 窗口能量积分（mJ）
    val acceptanceRate: Float = 0f  // 恒 0：无推测解码时不展示（UI 按 >0 隐藏），v0.5 接真实统计
)

/** v0.4 §5.3：多轮演示逐轮统计（会话终结即清空） */
data class TurnStat(val turn: Int, val ttftMs: Float, val totalTokens: Int)

data class BenchmarkUiState(
    val isRunning: Boolean = false,
    val currentPrompt: String = "",
    val generatedText: String = "",
    val result: BenchmarkResult? = null,
    val logs: List<String> = emptyList(),
    val error: String? = null,
    val hardwareInfo: String = "",
    val modelLoaded: Boolean = false,
    val turns: List<TurnStat> = emptyList(),
    val recommendation: String = ""
)

class BenchmarkViewModel : ViewModel() {

    var uiState by mutableStateOf(BenchmarkUiState())
        private set

    // v0.4 §6：应用推荐的透传旗标（null=默认路径，与 v0.3 全等）；一次性、下次 loadModel 消费
    private var pendingThreads: Int? = null
    private var pendingKvBits: Int? = null

    fun detectHardware() {
        viewModelScope.launch {
            try {
                val info = withContext(Dispatchers.Default) {
                    NativeEngine.getHardwareInfo()
                }
                uiState = uiState.copy(hardwareInfo = info)
                addLog("硬件检测完成")
            } catch (e: Exception) {
                uiState = uiState.copy(error = "硬件检测失败: ${e.message}")
            }
        }
    }

    fun loadModel(modelPath: String) {
        viewModelScope.launch {
            uiState = uiState.copy(isRunning = true, error = null)
            addLog("加载模型: $modelPath")

            val applied = pendingThreads != null || pendingKvBits != null
            try {
                val success = withContext(Dispatchers.Default) {
                    // 未点应用 → pending 恒 null → init(path, 0, 16) 与 v0.3 全等（brief Step 4 回归自查）
                    NativeEngine.init(modelPath, pendingThreads ?: 0, pendingKvBits ?: 16)
                }
                if (success) {
                    // 重载 = 新会话（native KV 重装），演示轮次随之清零（§5.3）
                    uiState = uiState.copy(modelLoaded = true, isRunning = false, turns = emptyList())
                    addLog("模型加载成功" + if (applied) " (threads/kv 应用推荐)" else " (默认)")
                } else {
                    // 加载失败时 native 引擎必不可用（守卫已 reset）——旗标不许撒谎（I-1c）
                    uiState = uiState.copy(
                        modelLoaded = false,
                        error = "模型加载失败",
                        isRunning = false,
                        turns = emptyList()
                    )
                }
            } catch (e: Exception) {
                uiState = uiState.copy(
                    modelLoaded = false,
                    error = "异常: ${e.message}",
                    isRunning = false,
                    turns = emptyList()
                )
            }
        }
    }

    /** 引擎被外部释放（矩阵收尾）：重置加载旗标，单跑强制走全新加载=推荐配置（I-2 握手机制） */
    fun onEngineReleased() {
        uiState = uiState.copy(modelLoaded = false, turns = emptyList())
    }

    /** 开关拨关：演示轮次清零（native KV 不动，下次运行按新会话重装——§5.3） */
    fun endSessionDemo() { uiState = uiState.copy(turns = emptyList()) }

    /** v0.4 §6：拉取推荐 JSON（loaded 分支=实态+预算校验；unloaded=预测）。
     *  native 侧 detectHardware()+getKVCacheInfo 无 JNI_OnLoad 挂——不得在主线程直调（T9 评审 Minor-1） */
    fun fetchRecommendation() {
        viewModelScope.launch {
            val json = withContext(Dispatchers.Default) { NativeEngine.recommendation() }
            uiState = uiState.copy(recommendation = json)
        }
    }

    /** 应用=一次性旗标：下次 loadModel 透传推荐 threads/kv（spec §6，ctx 由 native 预算复核）。
     *  应用链语义（T4 评审带入注释）：同 kv 同线程再 apply → 下次 init 三段同值走 fast-path
     *  复用——等价配置无需重载，无害；配置有差即真重载（g_active_kv 参与 fast-path 第三段）。 */
    fun applyRecommendation() {
        val o = try { JSONObject(uiState.recommendation) } catch (e: Exception) { null }
        if (o == null || !o.optBoolean("loaded")) { addLog("推荐不可用（模型未加载），未应用"); return }
        pendingThreads = o.optInt("threads"); pendingKvBits = o.optInt("kv_bits")
        addLog("已应用推荐配置: threads=$pendingThreads kv=$pendingKvBits（下次加载生效）")
    }

    fun runBenchmark(prompt: String, maxTokens: Int = 128, continueSession: Boolean = false) {
        if (!uiState.modelLoaded) {
            uiState = uiState.copy(error = "模型未加载")
            return
        }
        if (uiState.isRunning) return

        uiState = uiState.copy(
            isRunning = true,
            currentPrompt = prompt,
            generatedText = "",
            result = null,
            error = null,
            turns = if (continueSession) uiState.turns else emptyList()  // 新会话起算清零（§5.3）
        )
        addLog("流式推理开始: prompt=${prompt.take(40)}…, maxTokens=$maxTokens" +
            if (continueSession) " (续写第 ${uiState.turns.size + 1} 轮)" else "")

        viewModelScope.launch {
            val done = CompletableDeferred<String?>()
            val listener = object : NativeEngine.StreamListener {
                override fun onToken(piece: String, tokenId: Int) {
                    viewModelScope.launch(Dispatchers.Main) {
                        uiState = uiState.copy(generatedText = uiState.generatedText + piece)
                    }
                }

                override fun onDone(resultJson: String) {
                    if (!done.isCompleted) done.complete(resultJson)
                }
            }

            NativeEngine.samplerBegin()
            val started = withContext(Dispatchers.Default) {
                NativeEngine.generateStream(prompt, maxTokens, listener, continueSession)
            }
            val windowJson = NativeEngine.samplerEnd()  // native 路径 onDone 先于返回，窗口已含全程
            if (!started) {
                uiState = uiState.copy(isRunning = false, error = "流式推理启动失败", turns = emptyList())
                return@launch
            }
            val resultJson = done.await()  // started ⇒ onDone 必达（native 收尾保证；Mock 见 R7）

            val output = NativeEngine.parseGenerateOutput(resultJson ?: "{}")
            // N4：prefill 失败 → 显式错误，绝不以 0 值假成功上屏（会话同判终结，§5.3）
            if (output.totalTokens == 0 && output.text.startsWith("[错误]")) {
                uiState = uiState.copy(isRunning = false, error = output.text, turns = emptyList())
                addLog("推理失败: ${output.text}")
                return@launch
            }

            val agg = SamplerWindow.parse(windowJson)
            val expected = (output.totalTokens - 1).coerceAtLeast(0)
            if (output.itlSeries.size != expected)
                addLog("警告: ITL序列 ${output.itlSeries.size} != 预期 $expected")
            uiState = uiState.copy(
                isRunning = false,
                result = BenchmarkResult(
                    ttftMs = output.ttftMs,
                    itlAvgMs = output.itlAvgMs,
                    itlP50Ms = output.itlP50Ms,
                    itlP90Ms = output.itlP90Ms,
                    itlP99Ms = output.itlP99Ms,
                    tokensPerSec = output.tokensPerSec,
                    totalTokens = output.totalTokens,
                    totalTimeMs = output.totalTimeMs,
                    itlSeries = output.itlSeries,
                    powerTimeline = agg.powerMw,
                    energyMj = agg.energyMJ.toFloat()
                ),
                turns = uiState.turns + TurnStat(uiState.turns.size + 1, output.ttftMs, output.totalTokens)
            )
            addLog("推理完成: ${output.totalTokens} tokens, ${output.totalTimeMs.toInt()}ms, " +
                "TTFT ${output.ttftMs.toInt()}ms, ${String.format("%.1f", output.tokensPerSec)} tok/s" +
                " · 功耗采样 ${agg.nPower} 点, ${String.format(Locale.US, "%.2f", agg.energyMJ / 1000.0)}J")
        }
    }

    fun cancelRun() {
        if (uiState.isRunning) {
            NativeEngine.cancel()
            addLog("已请求停止生成")
        }
    }

    fun getMetrics() {
        viewModelScope.launch {
            try {
                val metrics = withContext(Dispatchers.Default) {
                    NativeEngine.getMetrics()
                }
                addLog("指标: $metrics")
            } catch (e: Exception) {
                addLog("获取指标失败: ${e.message}")
            }
        }
    }

    private fun addLog(message: String) {
        val timestamp = System.currentTimeMillis()
        uiState = uiState.copy(
            logs = uiState.logs + "[$timestamp] $message"
        )
    }

    fun clearLogs() {
        uiState = uiState.copy(logs = emptyList())
    }

    fun clearError() {
        uiState = uiState.copy(error = null)
    }

    override fun onCleared() {
        super.onCleared()
        NativeEngine.release()
    }
}
