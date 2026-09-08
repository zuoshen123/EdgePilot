package com.edgepilot.viewmodel

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.edgepilot.native.NativeEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class BenchmarkResult(
    val ttftMs: Float = 0f,
    val itlAvgMs: Float = 0f,
    val itlP99Ms: Float = 0f,
    val tokensPerSec: Float = 0f,
    val totalTokens: Int = 0,
    val totalTimeMs: Float = 0f,
    val acceptanceRate: Float = 0f
)

data class BenchmarkUiState(
    val isRunning: Boolean = false,
    val currentPrompt: String = "",
    val generatedText: String = "",
    val result: BenchmarkResult? = null,
    val logs: List<String> = emptyList(),
    val error: String? = null,
    val hardwareInfo: String = "",
    val modelLoaded: Boolean = false
)

class BenchmarkViewModel : ViewModel() {

    var uiState by mutableStateOf(BenchmarkUiState())
        private set

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

            try {
                val success = withContext(Dispatchers.Default) {
                    NativeEngine.init(modelPath)
                }
                if (success) {
                    uiState = uiState.copy(modelLoaded = true, isRunning = false)
                    addLog("模型加载成功")
                } else {
                    uiState = uiState.copy(
                        error = "模型加载失败",
                        isRunning = false
                    )
                }
            } catch (e: Exception) {
                uiState = uiState.copy(
                    error = "异常: ${e.message}",
                    isRunning = false
                )
            }
        }
    }

    fun runBenchmark(prompt: String, maxTokens: Int = 128) {
        if (!uiState.modelLoaded) {
            uiState = uiState.copy(error = "模型未加载")
            return
        }

        viewModelScope.launch {
            uiState = uiState.copy(
                isRunning = true,
                currentPrompt = prompt,
                generatedText = "",
                error = null
            )
            addLog("开始推理: prompt=$prompt, maxTokens=$maxTokens")

            try {
                val output = withContext(Dispatchers.Default) {
                    NativeEngine.generate(prompt, maxTokens)
                }

                // 计算平均 ITL
                val itlAvg = if (output.totalTokens > 1)
                    (output.totalTimeMs - output.ttftMs) / (output.totalTokens - 1) else 0f

                uiState = uiState.copy(
                    isRunning = false,
                    generatedText = output.text,
                    result = BenchmarkResult(
                        ttftMs = output.ttftMs,
                        itlAvgMs = itlAvg,
                        itlP99Ms = itlAvg * 1.2f,  // 简单估算 P99
                        totalTokens = output.totalTokens,
                        totalTimeMs = output.totalTimeMs,
                        tokensPerSec = output.tokensPerSec
                    )
                )
                addLog("推理完成: ${output.totalTokens} tokens, ${output.totalTimeMs.toInt()}ms, ${output.tokensPerSec} tok/s")
            } catch (e: Exception) {
                uiState = uiState.copy(
                    isRunning = false,
                    error = "推理失败: ${e.message}"
                )
            }
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
