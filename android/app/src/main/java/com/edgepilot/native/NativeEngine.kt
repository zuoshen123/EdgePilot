package com.edgepilot.native

import android.util.Log
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
    val totalTimeMs: Float = 0f
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
            // Mock 模式
            Log.i(TAG, "[Mock] 生成: prompt=${prompt.take(50)}...")
            GenerateOutput(
                text = "[Mock 模式] 这是 EdgePilot 的模拟回复。\n\n原始 prompt: ${prompt.take(100)}...",
                totalTokens = 20,
                tokensPerSec = 15.0f,
                ttftMs = 120f,
                totalTimeMs = 1400f
            )
        }
    }

    private fun parseGenerateOutput(json: String): GenerateOutput {
        return try {
            val obj = JSONObject(json)
            GenerateOutput(
                text = obj.optString("text", ""),
                totalTokens = obj.optInt("total_tokens", 0),
                tokensPerSec = obj.optDouble("tokens_per_sec", 0.0).toFloat(),
                ttftMs = obj.optDouble("ttft_ms", 0.0).toFloat(),
                totalTimeMs = obj.optDouble("total_time_ms", 0.0).toFloat()
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

    /**
     * 获取当前推理接受率 (用于 speculative decoding)
     */
    fun getAcceptanceRate(): Float {
        return if (nativeAvailable) {
            // 从指标中解析
            try {
                val metricsJson = nativeGetMetrics()
                // 简单解析 acceptance_rate 字段
                val regex = """"acceptance_rate"\s*:\s*([\d.]+)""".toRegex()
                regex.find(metricsJson)?.groupValues?.get(1)?.toFloatOrNull() ?: 0.7f
            } catch (e: Exception) {
                0.7f
            }
        } else {
            0.68f // Mock 接受率
        }
    }
}
