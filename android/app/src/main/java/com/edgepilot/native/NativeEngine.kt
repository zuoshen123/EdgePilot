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
    external fun nativeInit(modelPath: String): Boolean
    external fun nativeGetHardwareInfo(): String
    external fun nativeGenerate(prompt: String, maxTokens: Int): String
    external fun nativeGetMetrics(): String
    external fun nativeRelease()

    // ---- Kotlin 包装 (带 Mock 回退) ----

    /**
     * 初始化引擎
     * @param modelPath 模型文件路径 (如 /sdcard/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf)
     * @return 是否初始化成功
     */
    fun init(modelPath: String): Boolean {
        return if (nativeAvailable) {
            try {
                nativeInit(modelPath)
            } catch (e: Exception) {
                Log.e(TAG, "nativeInit 失败: ${e.message}")
                false
            }
        } else {
            Log.i(TAG, "[Mock] 初始化: $modelPath")
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
            try {
                val json = nativeGenerate(prompt, maxTokens)
                parseGenerateOutput(json)
            } catch (e: Exception) {
                Log.e(TAG, "generate 失败: ${e.message}")
                GenerateOutput(text = "[错误] 推理失败: ${e.message}")
            }
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

    /**
     * 释放资源
     */
    fun release() {
        if (nativeAvailable) {
            try {
                nativeRelease()
            } catch (e: Exception) {
                Log.e(TAG, "release 失败: ${e.message}")
            }
        }
    }
}
