#include <jni.h>
#include <string>
#include <memory>
#include <cstdio>
#include <cmath>

#include "edgepilot/backend/backend_factory.h"
#include "edgepilot/common/log.h"
#include "edgepilot/metrics/metrics_collector.h"

using namespace edgepilot;

static std::unique_ptr<InferenceBackend> g_backend;
static bool g_initialized = false;
static MetricsCollector g_metrics;         // v0.3 平台资源采样（spec §②）
static bool g_metrics_started = false;
static jint g_active_threads = 0;          // 上次成功加载所用的线程数
static int g_active_kv = 16;               // v0.4：当前生效 KV 量化档（fast-path 第三段 + kvInfo/换档同步）
static std::string g_model_path;           // 上次成功加载的模型路径

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
    JNIEnv* env, jobject, jstring modelPath, jint threads, jint kvBits)
{
    std::string path = jstring_to_string(env, modelPath);
    auto sane_kv = [](jint b) -> int {
        if (b == 16 || b == 8 || b == 4) return b;
        EP_LOGW("nativeInit: 非法 kvBits=%d → 回 16", (int)b); return 16;
    };
    const int kvb = sane_kv(kvBits);
    const bool same_path = (path == g_model_path);
    const bool same_threads = (threads <= 0) ? (g_active_threads > 0)
                                             : (threads == g_active_threads);
    const bool same_kv = (kvb == g_active_kv);
    if (g_initialized && same_path && same_threads && same_kv) {
        return JNI_TRUE;  // 快速复用：矩阵换 prompt 不重载模型（v0.4：换档必重载，量化对比依赖此）
    }
    if (g_initialized) {  // 换模型/线程/量化档：先卸再载（spec §②，矩阵逐线程列重载）
        EP_LOGI("nativeInit: 重载 (path_same=%d threads %d->%d kv %d->%d)",
                same_path ? 1 : 0, g_active_threads, (int)threads, g_active_kv, kvb);
        if (g_backend) { g_backend->unload(); g_backend.reset(); }
        g_initialized = false;
    }

    EP_LOGI("nativeInit: 初始化 EdgePilot (threads=%d kv=%d)", (int)threads, kvb);
    try {
        g_backend = BackendFactory::createOptimal();
        if (!g_backend) { EP_LOGE("创建后端失败"); return JNI_FALSE; }

        HardwareInfo hw = BackendFactory::detectHardware();
        ModelConfig cfg = BackendFactory::getRecommendedConfig(hw);
        cfg.model_path = path;
        if (threads > 0) cfg.threads = threads;
        cfg.kv_cache_bits = kvb;   // v0.4 §4：覆盖推荐配置的 16 默认（getRecommendedConfig 恒置 16）

        EP_LOGI("加载模型: %s (threads=%d kv=%d)", cfg.model_path.c_str(), cfg.threads, kvb);
        Status s = g_backend->loadModel(cfg);
        if (s != Status::OK) {
            EP_LOGE("模型加载失败: %s", statusToString(s));
            g_backend.reset();
            return JNI_FALSE;
        }

        g_active_threads = cfg.threads;
        g_active_kv = kvb;
        g_model_path = path;
        g_initialized = true;

        if (!g_metrics_started) {  // 采集线程全程常驻（spec §②），幂等
            MetricsCollector::Config mc{};
            mc.sample_interval_ms = 100;
            mc.enable_sqlite = false;
            mc.db_path = "";
            mc.enable_realtime_callback = false;
            mc.max_history_size = 100;
            g_metrics.initialize(mc);
            g_metrics.start();
            g_metrics_started = true;
        }
        EP_LOGI("初始化完成");
        return JNI_TRUE;
    } catch (const std::exception& e) {
        EP_LOGE("异常: %s", e.what());
        g_backend.reset();  // 不留"半构造"后端（M-3）；生成守卫由此走 g_backend==null → 显式异常
        g_initialized = false;   // nano-N6 收账：异常后不得残留 true（否则 fast-path 谎报可用）
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

JNIEXPORT jboolean JNICALL
Java_com_edgepilot_native_NativeEngine_nativeSessionSave(
    JNIEnv* env, jobject, jstring base)
{
    if (!g_initialized || !g_backend) { EP_LOGE("nativeSessionSave: 未初始化"); return JNI_FALSE; }
    return g_backend->saveSessionFile(jstring_to_string(env, base)) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_edgepilot_native_NativeEngine_nativeSessionLoad(
    JNIEnv* env, jobject, jstring base)
{
    if (!g_initialized || !g_backend) { EP_LOGE("nativeSessionLoad: 未初始化"); return JNI_FALSE; }
    return g_backend->loadSessionFile(jstring_to_string(env, base)) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_edgepilot_native_NativeEngine_nativeSetKvQuant(JNIEnv*, jobject, jint bits)
{
    if (!g_initialized || !g_backend) { EP_LOGE("nativeSetKvQuant: 未初始化"); return JNI_FALSE; }
    if (!g_backend->compressKVCache(bits)) {
        // T4 评审 Important 收账：重建失败后 core ctx 已死（model 在、bits 已前推），
        // fast-path 无从知晓——整体清场，让下一次 init 真重载，而非谎报 TRUE 直到 isLoaded 拦截
        EP_LOGW("nativeSetKvQuant: 重建失败——清场，下次 init 重载恢复");
        g_backend->unload(); g_backend.reset();
        g_initialized = false; g_model_path.clear(); g_active_threads = 0; g_active_kv = 16;
        return JNI_FALSE;
    }
    g_active_kv = bits;   // 与 core 侧 config 同步，防 kvInfo/nativeInit fast-path 说谎
    return JNI_TRUE;
}

JNIEXPORT jstring JNICALL
Java_com_edgepilot_native_NativeEngine_nativeKVInfoJson(JNIEnv* env, jobject)
{
    std::string j = "{\"used_bytes\":0,\"total_bytes\":0,\"n_ctx\":0,\"kv_bits\":" + std::to_string(g_active_kv) + "}";
    if (g_initialized && g_backend) {
        auto i = g_backend->getKVCacheInfo();
        j = "{\"used_bytes\":" + std::to_string(i.used_memory_bytes) +
            ",\"total_bytes\":" + std::to_string(i.total_memory_bytes) +
            ",\"n_ctx\":" + std::to_string(i.max_seq_len) +
            ",\"kv_bits\":" + std::to_string(g_active_kv) + "}";
    }
    return string_to_jstring(env, j);
}

JNIEXPORT void JNICALL
Java_com_edgepilot_native_NativeEngine_nativeGenerateStream(
    JNIEnv* env, jobject, jstring prompt, jint maxTokens, jfloat temperature, jboolean continueSession, jobject listener)
{
    if (!g_initialized || !g_backend || !listener) {
        EP_LOGE("nativeGenerateStream: 未初始化或 listener 为空");
        // 静默返回会让 Kotlin 侧 ok=true 且 onDone 零达（await 永久楔死）——挂起异常使 ok=false
        jclass guard_exc = env->FindClass("java/lang/IllegalStateException");
        if (guard_exc) env->ThrowNew(guard_exc, "engine not initialized");
        return;
    }

    GenerateRequest req{};
    req.prompt = jstring_to_string(env, prompt);
    req.max_new_tokens = maxTokens;
    req.temperature = (float)temperature;          // ≤0 → core 贪心（Task 1）；v0.3 传 0.8 恒非贪心
    req.continue_session = continueSession == JNI_TRUE;
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
Java_com_edgepilot_native_NativeEngine_nativeSamplerProbe(JNIEnv* env, jobject)
{
    return string_to_jstring(env, g_metrics.samplerProbeJson());
}

JNIEXPORT void JNICALL
Java_com_edgepilot_native_NativeEngine_nativeSamplerBegin(JNIEnv*, jobject)
{
    g_metrics.beginWindow();
}

JNIEXPORT jstring JNICALL
Java_com_edgepilot_native_NativeEngine_nativeSamplerEnd(JNIEnv* env, jobject)
{
    return string_to_jstring(env, g_metrics.endWindowJson());
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
    g_model_path.clear();
    g_active_threads = 0;
    g_active_kv = 16;   // v0.4：引擎消亡，量化档回到默认（防跨 release 谎报 fast-path）
}

} // extern "C"
