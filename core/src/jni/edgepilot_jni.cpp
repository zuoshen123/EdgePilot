#include <jni.h>
#include <string>
#include <memory>
#include <android/log.h>

#include "edgepilot/backend/backend_factory.h"

#define TAG "EdgePilot-JNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

using namespace edgepilot;

static std::unique_ptr<InferenceBackend> g_backend;
static bool g_initialized = false;

static std::string jstring_to_string(JNIEnv* env, jstring jstr) {
    if (!jstr) return "";
    const char* chars = env->GetStringUTFChars(jstr, nullptr);
    std::string result(chars);
    env->ReleaseStringUTFChars(jstr, chars);
    return result;
}

static jstring string_to_jstring(JNIEnv* env, const std::string& str) {
    return env->NewStringUTF(str.c_str());
}

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_edgepilot_native_NativeEngine_nativeInit(
    JNIEnv* env, jobject, jstring modelPath)
{
    if (g_initialized) return JNI_TRUE;

    LOGI("nativeInit: 初始化 EdgePilot");
    try {
        g_backend = BackendFactory::createOptimal();
        if (!g_backend) { LOGE("创建后端失败"); return JNI_FALSE; }

        HardwareInfo hw = BackendFactory::detectHardware();
        ModelConfig cfg = BackendFactory::getRecommendedConfig(hw);
        cfg.model_path = jstring_to_string(env, modelPath);

        LOGI("加载模型: %s", cfg.model_path.c_str());
        Status s = g_backend->loadModel(cfg);
        if (s != Status::OK) {
            LOGE("模型加载失败: %s", statusToString(s));
            return JNI_FALSE;
        }

        g_initialized = true;
        LOGI("初始化完成");
        return JNI_TRUE;
    } catch (const std::exception& e) {
        LOGE("异常: %s", e.what());
        return JNI_FALSE;
    }
}

JNIEXPORT jstring JNICALL
Java_com_edgepilot_native_NativeEngine_nativeGetHardwareInfo(JNIEnv* env, jobject)
{
    if (!g_backend) return string_to_jstring(env, "未初始化");

    HardwareInfo hw = g_backend->getHardwareInfo();
    std::string r;
    r += "SoC: " + hw.soc_name + "\n";
    r += "RAM: " + std::to_string(hw.total_memory_bytes / (1024*1024)) + "MB"
         + " (可用: " + std::to_string(hw.available_memory_bytes / (1024*1024)) + "MB)\n";
    r += "GPU: " + hw.gpu_name + "\n";
    r += "后端: " + g_backend->getName();
    return string_to_jstring(env, r);
}

JNIEXPORT jstring JNICALL
Java_com_edgepilot_native_NativeEngine_nativeGenerate(
    JNIEnv* env, jobject, jstring prompt, jint maxTokens)
{
    if (!g_initialized || !g_backend)
        return string_to_jstring(env, "[错误] 未初始化");

    std::string p = jstring_to_string(env, prompt);
    LOGI("生成: '%s', max=%d", p.c_str(), maxTokens);

    GenerateRequest req{};
    req.prompt = p;
    req.max_new_tokens = maxTokens;
    req.temperature = 0.8f;
    req.top_k = 40;
    req.top_p = 0.95f;

    GenerateResult res = g_backend->generate(req);

    LOGI("完成: %d tokens, %.1f tok/s, TTFT=%.1fms",
         res.total_tokens, res.tokens_per_sec, res.ttft_ms);

    // 返回 JSON：包含文本和指标
    std::string json = "{\"text\":";
    // 简单转义 JSON 字符串
    std::string escaped;
    for (char c : res.generated_text) {
        if (c == '"') escaped += "\\\"";
        else if (c == '\\') escaped += "\\\\";
        else if (c == '\n') escaped += "\\n";
        else escaped += c;
    }
    json += "\"" + escaped + "\"";
    json += ",\"total_tokens\":" + std::to_string(res.total_tokens);
    json += ",\"tokens_per_sec\":" + std::to_string(res.tokens_per_sec);
    json += ",\"ttft_ms\":" + std::to_string(res.ttft_ms);
    json += ",\"total_time_ms\":" + std::to_string(res.total_time_ms);
    json += "}";

    return string_to_jstring(env, json);
}

JNIEXPORT jstring JNICALL
Java_com_edgepilot_native_NativeEngine_nativeGetMetrics(JNIEnv* env, jobject)
{
    if (!g_backend) return string_to_jstring(env, "{}");

    KVCacheInfo kv = g_backend->getKVCacheInfo();
    std::string j = "{\"backend\":\"" + g_backend->getName()
        + "\",\"kv\":{\"used\":" + std::to_string(kv.used_memory_bytes)
        + ",\"max_seq\":" + std::to_string(kv.max_seq_len) + "}}";
    return string_to_jstring(env, j);
}

JNIEXPORT void JNICALL
Java_com_edgepilot_native_NativeEngine_nativeRelease(JNIEnv*, jobject)
{
    LOGI("释放引擎");
    if (g_backend) { g_backend->unload(); g_backend.reset(); }
    g_initialized = false;
}

} // extern "C"
