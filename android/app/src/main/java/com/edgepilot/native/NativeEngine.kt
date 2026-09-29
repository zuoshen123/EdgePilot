package com.edgepilot.native

import android.util.Log
import org.json.JSONArray
import org.json.JSONObject

/**
 * EdgePilot 原生引擎 — JNI 桥接层
 *
 * 调用 C++ 核心进行 LLM 推理。
 * 如果 native library 加载失败，回退到 Mock 模式。
 */

/** 推理结果，包含文本和性能指标 */
data class GenerateOutput(
    val text: String,
    val totalTokens: Int = 0,
    val tokensPerSec: Float = 0f,
    val ttftMs: Float = 0f,
    val totalTimeMs: Float = 0f,
    val itlAvgMs: Float = 0f,
    val itlP50Ms: Float = 0f,
    val itlP90Ms: Float = 0f,
    val itlP99Ms: Float = 0f,
    val itlSeries: List<Double> = emptyList()
)

object NativeEngine {
    private const val TAG = "NativeEngine"

    // 是否使用真实 native 引擎
    private var nativeAvailable = false

    init {
        try {
            System.loadLibrary("edgepilot_jni")
            nativeAvailable = true
            Log.i(TAG, "Native library 加载成功")
        } catch (e: UnsatisfiedLinkError) {
            Log.w(TAG, "Native library 加载失败，使用 Mock 模式: ${e.message}")
            nativeAvailable = false
        }
    }

    // ---- Native 方法声明 ----
    external fun nativeInit(modelPath: String, threads: Int): Boolean
    external fun nativeGetHardwareInfo(): String
    external fun nativeGenerate(prompt: String, maxTokens: Int): String

    /** 流式生成回调（在 native 调用线程触发，实现方自行切主线程） */
    interface StreamListener {
        fun onToken(piece: String, tokenId: Int)
        fun onDone(resultJson: String)
    }

    external fun nativeGenerateStream(prompt: String, maxTokens: Int, listener: Any)
    external fun nativeCancel()

    external fun nativeSamplerProbe(): String
    external fun nativeSamplerBegin()
    external fun nativeSamplerEnd(): String
    external fun nativeGetMetrics(): String
    external fun nativeRelease()

    // ---- Kotlin 包装 (带 Mock 回退) ----

    // N2: 生成在途时的延后释放（onCleared 不得在 native 生成线程使用引擎期间 reset 后端）
    @Volatile private var generating = false
    @Volatile private var releasePending = false

    private const val MOCK_CAPABILITY_JSON =
        """{"power":"ABSENT","thermal":"ABSENT","mem":"ABSENT","cpu":"ABSENT","power_path":"","zones":[]}"""
    private const val EMPTY_WINDOW_JSON =
        """{"power":[],"thermal":[],"mem":[],"n":{"power":0,"thermal":0,"mem":0}}"""

    /**
     * 初始化/重载引擎。同路径同线程数时快速复用（native 侧判断）。
     * @param threads CPU 线程数，<=0 用推荐配置
     */
    fun init(modelPath: String, threads: Int = 0): Boolean {
        return if (nativeAvailable) {
            try {
                nativeInit(modelPath, threads)
            } catch (e: Exception) {
                Log.e(TAG, "nativeInit 失败: ${e.message}")
                false
            }
        } else {
            Log.i(TAG, "[Mock] 初始化: $modelPath (threads=$threads)")
            true
        }
    }

    /**
     * 获取硬件信息
     */
    fun getHardwareInfo(): String {
        return if (nativeAvailable) {
            try {
                nativeGetHardwareInfo()
            } catch (e: Exception) {
                Log.e(TAG, "getHardwareInfo 失败: ${e.message}")
                "Error: ${e.message}"
            }
        } else {
            // Mock 模式
            """
            CPU: Snapdragon 8 Gen3 (Mock)
            RAM: 12288MB (可用: 8192MB)
            GPU: Adreno 750 (Mock)
            NPU: Hexagon (Mock)
            后端: Mock Backend
            """.trimIndent()
        }
    }

    /**
     * 文本生成（返回文本 + 性能指标）
     */
    fun generate(prompt: String, maxTokens: Int = 256): GenerateOutput {
        return if (nativeAvailable) {
            var ok: GenerateOutput
            generating = true
            try {
                val json = nativeGenerate(prompt, maxTokens)
                ok = parseGenerateOutput(json)
            } catch (e: Exception) {
                Log.e(TAG, "generate 失败: ${e.message}")
                ok = GenerateOutput(text = "[错误] 推理失败: ${e.message}")
            } finally {
                generating = false
                drainPendingRelease()
            }
            ok
        } else {
            // Mock 模式（native 库缺失时的兜底，输出文本带 [Mock] 标记可辨识）
            Log.i(TAG, "[Mock] 生成: prompt=${prompt.take(50)}...")
            val n = 20
            val per = 1400.0 / n
            GenerateOutput(
                text = "[Mock 模式] 这是 EdgePilot 的模拟回复。\n\n原始 prompt: ${prompt.take(100)}...",
                totalTokens = n,
                tokensPerSec = n / 1.4f,
                ttftMs = 120f,
                totalTimeMs = 1400f,
                itlAvgMs = per.toFloat(),
                itlP50Ms = per.toFloat(),
                itlP90Ms = (per * 1.1).toFloat(),
                itlP99Ms = (per * 1.2).toFloat(),
                itlSeries = List(n - 1) { per }
            )
        }
    }

    /**
     * 流式生成：逐 token 回调，结束后 onDone 携带完整指标 JSON。
     * 调用线程即回调线程，会阻塞到生成结束（应在后台线程调用）。
     * @return 是否成功启动（false = 未初始化/mock 环境失败）
     */
    fun generateStream(prompt: String, maxTokens: Int, listener: StreamListener): Boolean {
        return if (nativeAvailable) {
            var ok = false
            generating = true
            try {
                nativeGenerateStream(prompt, maxTokens, listener)
                ok = true
            } catch (e: Exception) {
                Log.e(TAG, "generateStream 失败: ${e.message}", e)
            } finally {
                generating = false
                drainPendingRelease()
            }
            ok
        } else {
            // Mock 流式：逐词吐出，模拟打字机
            Thread {
                "[Mock 流式] 这是 EdgePilot 的模拟逐 token 输出。".split(" ").forEachIndexed { i, w ->
                    listener.onToken(w + " ", i)
                    Thread.sleep(80)
                }
                listener.onDone("{\"text\":\"[Mock 流式] 这是 EdgePilot 的模拟逐 token 输出。\"," +
                    "\"total_tokens\":10,\"tokens_per_sec\":12.5,\"ttft_ms\":120," +
                    "\"total_time_ms\":800,\"itl_avg_ms\":75.5,\"itl_p50_ms\":75," +
                    "\"itl_p90_ms\":90,\"itl_p99_ms\":95,\"itl_series\":[80,75,70,75,80,75,70,75,75]}")
            }.start()
            true
        }
    }

    fun cancel() {
        if (nativeAvailable) {
            try {
                nativeCancel()
            } catch (e: Exception) {
                Log.e(TAG, "cancel 失败: ${e.message}")
            }
        }
    }

    fun parseGenerateOutput(json: String): GenerateOutput {
        return try {
            val obj = JSONObject(json)
            val seriesArr = obj.optJSONArray("itl_series") ?: JSONArray()
            val series = ArrayList<Double>(seriesArr.length())
            for (i in 0 until seriesArr.length()) series.add(seriesArr.getDouble(i))
            GenerateOutput(
                text = obj.optString("text", ""),
                totalTokens = obj.optInt("total_tokens", 0),
                tokensPerSec = obj.optDouble("tokens_per_sec", 0.0).toFloat(),
                ttftMs = obj.optDouble("ttft_ms", 0.0).toFloat(),
                totalTimeMs = obj.optDouble("total_time_ms", 0.0).toFloat(),
                itlAvgMs = obj.optDouble("itl_avg_ms", 0.0).toFloat(),
                itlP50Ms = obj.optDouble("itl_p50_ms", 0.0).toFloat(),
                itlP90Ms = obj.optDouble("itl_p90_ms", 0.0).toFloat(),
                itlP99Ms = obj.optDouble("itl_p99_ms", 0.0).toFloat(),
                itlSeries = series
            )
        } catch (e: Exception) {
            Log.e(TAG, "解析 JSON 失败: $json", e)
            GenerateOutput(text = json)
        }
    }

    /**
     * 获取性能指标
     */
    fun getMetrics(): String {
        return if (nativeAvailable) {
            try {
                nativeGetMetrics()
            } catch (e: Exception) {
                Log.e(TAG, "getMetrics 失败: ${e.message}")
                "{}"
            }
        } else {
            """
            {
                "backend": "Mock Backend",
                "initialized": true,
                "kv_cache": {"used": 0, "total": 2048}
            }
            """.trimIndent()
        }
    }

    /** 能力矩阵 JSON（spec §②；Mock 环境全 ABSENT——不伪造） */
    fun samplerProbe(): String {
        return if (nativeAvailable) {
            try { nativeSamplerProbe() } catch (e: Exception) {
                Log.e(TAG, "samplerProbe 失败: ${e.message}"); MOCK_CAPABILITY_JSON
            }
        } else MOCK_CAPABILITY_JSON
    }

    fun samplerBegin() {
        if (!nativeAvailable) return
        try { nativeSamplerBegin() } catch (e: Exception) { Log.e(TAG, "samplerBegin 失败: ${e.message}") }
    }

    /** 关窗导出窗口时间线 JSON；异常返回空窗（调用方按"无数据"处理，不伪造） */
    fun samplerEnd(): String {
        return if (nativeAvailable) {
            try { nativeSamplerEnd() } catch (e: Exception) {
                Log.e(TAG, "samplerEnd 失败: ${e.message}"); EMPTY_WINDOW_JSON
            }
        } else EMPTY_WINDOW_JSON
    }

    /**
     * 释放资源。生成在途时置 releasePending，由在途线程收尾时真正释放（N2）。
     * 注：release 与"下一次生成开始"同线程（Main/Default 串行）时窗口闭合；
     * 矩阵 runner 与 UI 单跑互斥使用引擎，不在 release 挂起时并发开新生成。
     */
    fun release() {
        if (!nativeAvailable) return
        if (generating) {
            releasePending = true
            Log.i(TAG, "release 延后至在途生成结束")
            return
        }
        try {
            nativeRelease()
        } catch (e: Exception) {
            Log.e(TAG, "release 失败: ${e.message}")
        }
    }

    private fun drainPendingRelease() {
        if (releasePending) {
            releasePending = false
            try {
                nativeRelease()
            } catch (e: Exception) {
                Log.e(TAG, "延后 release 失败: ${e.message}")
            }
        }
    }
}
