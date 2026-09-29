#include <jni.h>
#include <string>
#include <memory>
#include <cstdio>
#include <cmath>

#include "edgepilot/backend/backend_factory.h"
#include "edgepilot/common/log.h"

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

// 字符串 → JSON 安全内容（不含首尾引号）
static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char u[8];
                    std::snprintf(u, sizeof(u), "\\u%04x", c);
                    out += u;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

static std::string json_num(double v) {
    return std::isfinite(v) ? std::to_string(v) : std::string("0");
}

// GenerateResult → 指标 JSON（nativeGenerate 返回值与流式 onDone 参数共用）
static std::string metrics_to_json(const GenerateResult& r) {
    std::string j = "{\"text\":\"" + json_escape(r.generated_text) + "\"";
    j += ",\"total_tokens\":" + std::to_string(r.total_tokens);
    j += ",\"tokens_per_sec\":" + json_num(r.tokens_per_sec);
    j += ",\"ttft_ms\":" + json_num(r.ttft_ms);
    j += ",\"total_time_ms\":" + json_num(r.total_time_ms);
    j += ",\"itl_avg_ms\":" + json_num(r.itl_avg_ms);
    j += ",\"itl_p50_ms\":" + json_num(r.itl_p50_ms);
    j += ",\"itl_p90_ms\":" + json_num(r.itl_p90_ms);
    j += ",\"itl_p99_ms\":" + json_num(r.itl_p99_ms);
    j += ",\"itl_series\":[";
    for (size_t i = 0; i < r.itl_series_ms.size(); ++i) {
        if (i) j += ",";
        j += json_num(r.itl_series_ms[i]);
    }
    j += "]}";
    return j;
}

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_edgepilot_native_NativeEngine_nativeInit(
    JNIEnv* env, jobject, jstring modelPath)
{
    if (g_initialized) return JNI_TRUE;

    EP_LOGI("nativeInit: 初始化 EdgePilot");
    try {
        g_backend = BackendFactory::createOptimal();
        if (!g_backend) { EP_LOGE("创建后端失败"); return JNI_FALSE; }

        HardwareInfo hw = BackendFactory::detectHardware();
        ModelConfig cfg = BackendFactory::getRecommendedConfig(hw);
        cfg.model_path = jstring_to_string(env, modelPath);

        EP_LOGI("加载模型: %s", cfg.model_path.c_str());
        Status s = g_backend->loadModel(cfg);
        if (s != Status::OK) {
            EP_LOGE("模型加载失败: %s", statusToString(s));
            return JNI_FALSE;
        }

        g_initialized = true;
        EP_LOGI("初始化完成");
        return JNI_TRUE;
    } catch (const std::exception& e) {
        EP_LOGE("异常: %s", e.what());
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
    EP_LOGI("生成: '%s', max=%d", p.c_str(), maxTokens);

    GenerateRequest req{};
    req.prompt = p;
    req.max_new_tokens = maxTokens;
    req.temperature = 0.8f;
    req.top_k = 40;
    req.top_p = 0.95f;

    GenerateResult res = g_backend->generate(req);

    EP_LOGI("完成: %d tokens, %.1f tok/s, TTFT=%.1fms",
         res.total_tokens, res.tokens_per_sec, res.ttft_ms);

    return string_to_jstring(env, metrics_to_json(res));
}

JNIEXPORT void JNICALL
Java_com_edgepilot_native_NativeEngine_nativeGenerateStream(
    JNIEnv* env, jobject, jstring prompt, jint maxTokens, jobject listener)
{
    if (!g_initialized || !g_backend || !listener) {
        EP_LOGE("nativeGenerateStream: 未初始化或 listener 为空");
        return;
    }

    GenerateRequest req{};
    req.prompt = jstring_to_string(env, prompt);
    req.max_new_tokens = maxTokens;
    req.temperature = 0.8f;
    req.top_k = 40;
    req.top_p = 0.95f;

    jclass cls = env->GetObjectClass(listener);
    jmethodID onToken = env->GetMethodID(cls, "onToken", "(Ljava/lang/String;I)V");
    jmethodID onDone  = env->GetMethodID(cls, "onDone",  "(Ljava/lang/String;)V");
    if (!onToken || !onDone) {
        EP_LOGE("listener 方法签名不匹配: 需要 onToken(String,Int) / onDone(String)");
        return;
    }
    jobject lref = env->NewGlobalRef(listener);

    // 生成循环同步运行在本（JNI 调用）线程，env 全程有效
    g_backend->generateAsync(req, [&](const TokenResult& tr) {
        jstring piece = env->NewStringUTF(tr.token_text.c_str());
        env->CallVoidMethod(lref, onToken, piece, static_cast<jint>(tr.token_id));
        env->DeleteLocalRef(piece);
        if (env->ExceptionCheck()) {
            env->ExceptionDescribe();
            env->ExceptionClear();
        }
    });

    std::string json = metrics_to_json(g_backend->getLastMetrics());
    jstring jres = env->NewStringUTF(json.c_str());
    env->CallVoidMethod(lref, onDone, jres);
    env->DeleteLocalRef(jres);
    if (env->ExceptionCheck()) {
        env->ExceptionDescribe();
        env->ExceptionClear();
    }
    env->DeleteGlobalRef(lref);
}

JNIEXPORT void JNICALL
Java_com_edgepilot_native_NativeEngine_nativeCancel(JNIEnv*, jobject)
{
    if (g_backend) {
        EP_LOGI("cancel requested");
        g_backend->cancel();
    }
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
    EP_LOGI("释放引擎");
    if (g_backend) { g_backend->unload(); g_backend.reset(); }
    g_initialized = false;
}

} // extern "C"
