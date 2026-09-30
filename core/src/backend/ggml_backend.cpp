#include "edgepilot/backend/ggml_backend.h"

#include "llama.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <vector>

#include "edgepilot/common/log.h"

#ifdef __ANDROID__
#include <sys/sysinfo.h>
#endif

namespace edgepilot {

// 流式/同步早退共用：失败必须让用户看见（N4，spec §⑤）
static constexpr const char* kPrefillError = "[错误] prefill 失败（可能超出上下文）";

// v0.4 §2：sidecar helper（自家写、自家读，字段固定，不引 JSON 库）
namespace {
std::string json_escape_basic(const std::string& s) {   // sidecar 只可能踩双引号与反斜杠（原文 "\ 结尾会行接续吞声明，改述）
    std::string o; o.reserve(s.size() + 4);
    for (char c : s) { if (c=='"'||c=='\\') { o+='\\'; o+=c; } else if ((unsigned char)c < 0x20) o += ' '; else o += c; }
    return o;
}
bool sidecar_fetch(const std::string& j, const std::string& key, std::string& out) {
    // 手写取值：仅支持 "key":"str" 与 "key":number 两型
    std::string pat = "\"" + key + "\":";
    size_t p = j.find(pat); if (p == std::string::npos) return false;
    p += pat.size();
    if (p >= j.size()) return false;   // 键在尾部无值 → 视为非法（防御，自写文件永不触发）
    if (j[p] == '"') { size_t e = j.find('"', p+1); if (e==std::string::npos) return false; out = j.substr(p+1, e-p-1); return true; }
    size_t e = j.find_first_of(",}", p); if (e==std::string::npos) e = j.size();
    out = j.substr(p, e-p); return true;
}
long long wall_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}
} // namespace

// v0.4 §4：量化档位 → ggml 类型（load/compress/info 三处共用，禁复制散布）。K 保守 Q8（用户裁定④）。
static void kv_types(int bits, ggml_type& k, ggml_type& v) {
    switch (bits) {
        case 8:  k = GGML_TYPE_Q8_0; v = GGML_TYPE_Q8_0; break;
        case 4:  k = GGML_TYPE_Q8_0; v = GGML_TYPE_Q4_0; break;
        default: k = GGML_TYPE_F16;  v = GGML_TYPE_F16;  break;
    }
}

// loadModel 与 compressKVCache 共用的 ctx 参数组装（v0.4：type_k/v 自 kv_cache_bits 生效）
static llama_context_params make_cparams(const ModelConfig& config) {
    auto cp = llama_context_default_params();
    cp.n_ctx   = config.context_length > 0 ? config.context_length : 2048;
    cp.n_batch = config.batch_size > 0 ? config.batch_size : 512;
    cp.n_threads = config.threads > 0 ? config.threads
                        : std::min(static_cast<int32_t>(std::thread::hardware_concurrency()), 4);
    cp.n_threads_batch = cp.n_threads;
    ggml_type tk, tv; kv_types(config.kv_cache_bits, tk, tv);
    cp.type_k = tk; cp.type_v = tv;
    return cp;
}

// nearest-rank 分位数：idx = ceil(q*n) - 1（升序数组）
static double percentile_nearest_rank(const std::vector<double>& sorted, double q) {
    if (sorted.empty()) return 0.0;
    size_t idx = static_cast<size_t>(std::ceil(q * static_cast<double>(sorted.size())));
    if (idx == 0) idx = 1;
    if (idx > sorted.size()) idx = sorted.size();
    return sorted[idx - 1];
}

// 由逐 token 生成时刻序列计算 ITL 统计（序列本身 + avg/P50/P90/P99）
static void compute_itl_metrics(GenerateResult& r, const std::vector<double>& token_times) {
    r.itl_series_ms.clear();
    for (size_t i = 1; i < token_times.size(); ++i)
        r.itl_series_ms.push_back(token_times[i] - token_times[i - 1]);

    if (r.itl_series_ms.empty()) {
        r.itl_avg_ms = r.itl_p50_ms = r.itl_p90_ms = r.itl_p99_ms = 0.0f;
        return;
    }
    double sum = 0.0;
    for (double v : r.itl_series_ms) sum += v;
    r.itl_avg_ms = static_cast<float>(sum / static_cast<double>(r.itl_series_ms.size()));

    auto sorted = r.itl_series_ms;
    std::sort(sorted.begin(), sorted.end());
    r.itl_p50_ms = static_cast<float>(percentile_nearest_rank(sorted, 0.50));
    r.itl_p90_ms = static_cast<float>(percentile_nearest_rank(sorted, 0.90));
    r.itl_p99_ms = static_cast<float>(percentile_nearest_rank(sorted, 0.99));
}

// ============================================================================
// GgmlBackend::Impl — 内部实现
// ============================================================================
struct GgmlBackend::Impl {
    // ---- llama.cpp 对象 ----
    llama_model*    model    = nullptr;
    llama_context*  ctx      = nullptr;
    llama_sampler*  sampler  = nullptr;
    const llama_vocab* vocab = nullptr;

    // ---- 配置 ----
    ModelConfig config;

    // ---- 生成状态 ----
    std::mutex  mtx;
    std::atomic<bool> cancelled{false};
    int32_t     n_past    = 0;    // KV cache 中已有的 token 数
    std::vector<llama_token> session_tokens;   // 当前会话 token 历史（save_file 必需；reset 清空）

    // ---- 指标 ----
    int64_t     total_tokens_generated = 0;
    double      total_generation_ms    = 0.0;
    std::vector<double> itl_history;
    GenerateResult last_metrics{};

    // ---- 工具方法 ----

    static double now_ms() {
        using namespace std::chrono;
        return duration<double, std::milli>(
            steady_clock::now().time_since_epoch()
        ).count();
    }

    llama_sampler* buildSampler(float temperature) {
        auto sparams = llama_sampler_chain_default_params();
        llama_sampler* chain = llama_sampler_chain_init(sparams);
        if (temperature <= 0.0f) {   // v0.4 §2.3 贪心前置：temp≤0 → 确定性采样（replay 一致性基础）
            llama_sampler_chain_add(chain, llama_sampler_init_greedy());
            return chain;
        }
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(40));
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(0.95, 1));
        llama_sampler_chain_add(chain, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(chain, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
        return chain;
    }

    // v0.4 §1.2：续写轮的角色胶水。模板缺失/不适用 → 显式退化文本（演示玩具级，测量真实）
    std::string chatGlue(const std::string& user_text) {
        const char* tmpl = model ? llama_model_chat_template(model, nullptr) : nullptr;
        if (tmpl) {
            llama_chat_message msg{ "user", user_text.c_str() };   // 本版 struct 仅 {role,content}（llama.h:461-464）
            std::vector<char> buf(2 * (user_text.size() + 64));
            int32_t n = llama_chat_apply_template(tmpl, &msg, 1, true, buf.data(), (int32_t)buf.size());
            if (n >= 0) {
                if (static_cast<size_t>(n) > buf.size()) {   // 模板输出超预估：按返回值扩容重放一次
                    buf.resize(n);
                    n = llama_chat_apply_template(tmpl, &msg, 1, true, buf.data(), n);
                    if (n >= 0) return std::string(buf.data(), n);
                } else {
                    return std::string(buf.data(), n);
                }
            }
        }
        return "\n用户: " + user_text + "\n助手:";
    }

    std::string tokenToPiece(llama_token token) {
        char buf[256];
        int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
        if (n <= 0) return "";
        return std::string(buf, n);
    }

    std::vector<llama_token> tokenize(const std::string& text, bool add_special = true) {
        int n = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()),
                               nullptr, 0, add_special, false);
        if (n == 0) return {};
        if (n < 0) n = -n;  // 负数表示需要的 token 数量
        std::vector<llama_token> tokens(n);
        llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()),
                       tokens.data(), n, add_special, false);
        return tokens;
    }

    void resetKVCache() {
        if (ctx) {
            llama_memory_clear(llama_get_memory(ctx), true);
            n_past = 0;
            session_tokens.clear();   // v0.4 §2.1：会话历史随 KV 一同终结
        }
    }
};

// ============================================================================
// 构造 / 析构
// ============================================================================
GgmlBackend::GgmlBackend() : impl_(std::make_unique<Impl>()) {
    llama_backend_init();
}

GgmlBackend::~GgmlBackend() {
    unload();
}

// ============================================================================
// loadModel
// ============================================================================
Status GgmlBackend::loadModel(const ModelConfig& config) {
    if (impl_->model) unload();

    impl_->config = config;
    state_ = InferenceState::LOADING;

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = config.gpu_layers;

    impl_->model = llama_model_load_from_file(config.model_path.c_str(), mparams);
    if (!impl_->model) {
        state_ = InferenceState::ERROR;
        return Status::MODEL_LOAD_FAILED;
    }

    impl_->vocab = llama_model_get_vocab(impl_->model);
    EP_LOGI("vocab=%p", (void*)impl_->vocab);

#ifdef __ANDROID__
    // v0.4 §6/P-ctx：KV 预算校验（按 F16 保守估）——超预算则 ctx 逐档降 4096→2048→1024→512
    // （0.55 系数 twin 口径见 edgepilot_jni.cpp nativeGetRecommendationJson 的 budget 式）
    {
        auto cp = make_cparams(config);
        const int32_t n_head_mdl   = llama_model_n_head(impl_->model);
        const int32_t head_dim_mdl = n_head_mdl > 0 ? llama_model_n_embd(impl_->model) / n_head_mdl : 0;  // 零除防御（T3 裁定①）
        const size_t per_tok_f16 = 2ULL * static_cast<size_t>(llama_model_n_layer(impl_->model))
            * static_cast<size_t>(llama_model_n_head_kv(impl_->model))
            * static_cast<size_t>(head_dim_mdl) * 2ULL;
        struct sysinfo si{};
        if (per_tok_f16 > 0 && sysinfo(&si) == 0) {
            size_t avail = si.freeram * si.mem_unit, budget = avail * 55 / 100;
            const int tiers[4] = {4096, 2048, 1024, 512};
            int pick = static_cast<int>(cp.n_ctx);
            for (int t : tiers) {
                size_t need = static_cast<size_t>(t) * per_tok_f16 * 115 / 100;   // §6 15% 裕量
                if (need <= budget) { pick = t < static_cast<int>(cp.n_ctx) ? t : static_cast<int>(cp.n_ctx); break; }
                pick = 512;   // 全超：封顶 512，卡片另行警示
            }
            if (pick < static_cast<int>(cp.n_ctx)) {
                EP_LOGI("KV 预算降档: n_ctx %d→%d (per_tok=%zuB budget=%zuMB)",
                        static_cast<int>(cp.n_ctx), pick, per_tok_f16, budget >> 20);
                cp.n_ctx = static_cast<uint32_t>(pick);
            }
            impl_->ctx = llama_init_from_model(impl_->model, cp);
        } else {
            impl_->ctx = llama_init_from_model(impl_->model, cp);
        }
    }
#else
    impl_->ctx = llama_init_from_model(impl_->model, make_cparams(config));
#endif
    if (!impl_->ctx) {
        llama_model_free(impl_->model);
        impl_->model = nullptr;
        state_ = InferenceState::ERROR;
        return Status::MODEL_LOAD_FAILED;
    }

    impl_->sampler = impl_->buildSampler(0.8f);   // v0.3 行为逐字不变（固定 0.8 采样链）
    impl_->n_past = 0;
    impl_->session_tokens.clear();   // 同上（loadModel 顶部 unload() 已覆盖重载路径，此处防御首载残留）
    impl_->total_tokens_generated = 0;
    impl_->total_generation_ms = 0.0;
    impl_->itl_history.clear();

    state_ = InferenceState::IDLE;
    return Status::OK;
}

void GgmlBackend::unload() {
    if (impl_->sampler) { llama_sampler_free(impl_->sampler); impl_->sampler = nullptr; }
    if (impl_->ctx)     { llama_free(impl_->ctx);              impl_->ctx = nullptr; }
    if (impl_->model)   { llama_model_free(impl_->model);      impl_->model = nullptr; }
    impl_->vocab = nullptr;
    impl_->n_past = 0;
    impl_->session_tokens.clear();   // v0.4 T1评审带入：KV 随 ctx 消亡，历史必须同清（防陈旧向量骗过 save 守卫）
    state_ = InferenceState::IDLE;
}

bool GgmlBackend::isLoaded() const {
    return impl_->model != nullptr && impl_->ctx != nullptr;
}

// ============================================================================
// generate — 同步生成
// ============================================================================
GenerateResult GgmlBackend::generate(const GenerateRequest& request) {
    GenerateResult result{};
    if (!isLoaded()) return result;

    state_ = InferenceState::DECODING;
    std::lock_guard<std::mutex> lock(impl_->mtx);
    impl_->cancelled = false;

    // v0.4：temp≤0 → 逐调用贪心链（即用即释），否则共享常驻链（v0.3 行为）
    llama_sampler* greedy = request.temperature <= 0.0f ? impl_->buildSampler(request.temperature) : nullptr;
    llama_sampler* sampler = greedy ? greedy : impl_->sampler;

    double gen_start = Impl::now_ms();
    std::vector<double> token_times;

    // Tokenize prompt（v0.4 §1.1：续写轮不清 KV、不加 BOS，仅提交胶水后的增量）
    const bool cont = request.continue_session && impl_->n_past > 0;
    if (request.continue_session && !cont)
        EP_LOGW("continue_session=true 但无会话态——回退全新装载（P-1）");
    std::vector<llama_token> prompt_tokens = cont
        ? impl_->tokenize(impl_->chatGlue(request.prompt), false)
        : impl_->tokenize(request.prompt, true);
    EP_LOGI("tokenize: %d tokens", (int)prompt_tokens.size());
    if (prompt_tokens.empty()) {
        if (greedy) llama_sampler_free(greedy);
        result.generated_text = kPrefillError;
        state_ = InferenceState::IDLE;
        return result;
    }

    if (!cont) impl_->resetKVCache();     // 非续写路径逐字等价（原无条件 reset）
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(),
                                            static_cast<int32_t>(prompt_tokens.size()));
    int prefill_ret = llama_decode(impl_->ctx, batch);
    EP_LOGI("prefill decode ret=%d (cont=%d, %d tokens)", prefill_ret, cont ? 1 : 0, (int)prompt_tokens.size());
    if (prefill_ret != 0) {
        if (greedy) llama_sampler_free(greedy);
        result.generated_text = kPrefillError;   // 续写失败不动旧会话（未 append）
        state_ = InferenceState::IDLE;
        return result;
    }
    impl_->n_past += static_cast<int32_t>(prompt_tokens.size());
    impl_->session_tokens.insert(impl_->session_tokens.end(),
                                 prompt_tokens.begin(), prompt_tokens.end());

    double first_token_time = 0.0;
    int32_t n_predict = request.max_new_tokens > 0 ? request.max_new_tokens : 256;
    llama_token new_token_id = 0;

    for (int32_t i = 0; i < n_predict; ++i) {
        if (impl_->cancelled) break;

        new_token_id = llama_sampler_sample(sampler, impl_->ctx, -1);
        if (i < 3) EP_LOGI("sample[%d] id=%d eog=%d", i, new_token_id, llama_vocab_is_eog(impl_->vocab, new_token_id));
        if (llama_vocab_is_eog(impl_->vocab, new_token_id)) break;

        if (i == 0) {
            first_token_time = Impl::now_ms();
            result.ttft_ms = static_cast<float>(first_token_time - gen_start);
        }

        std::string piece = impl_->tokenToPiece(new_token_id);
        result.generated_text += piece;
        result.generated_tokens.push_back(static_cast<int>(new_token_id));
        impl_->session_tokens.push_back(new_token_id);   // v0.4 §2.1：生成 token 入会话历史
        result.total_tokens++;
        token_times.push_back(Impl::now_ms());

        llama_batch next_batch = llama_batch_get_one(&new_token_id, 1);
        if (llama_decode(impl_->ctx, next_batch) != 0) break;
        impl_->n_past++;
    }

    double gen_end = Impl::now_ms();
    result.total_time_ms = static_cast<float>(gen_end - gen_start);
    if (result.total_tokens > 0 && result.total_time_ms > 0) {
        result.tokens_per_sec = result.total_tokens / (result.total_time_ms / 1000.0f);
    }

    compute_itl_metrics(result, token_times);
    impl_->last_metrics = result;   // generate() 顶部已全程持有 impl_->mtx，此处直接赋值、切勿再次加锁（非递归锁会死锁）

    impl_->total_tokens_generated += result.total_tokens;
    impl_->total_generation_ms += result.total_time_ms;

    if (greedy) llama_sampler_free(greedy);
    state_ = InferenceState::IDLE;
    return result;
}

// ============================================================================
// generateAsync — 流式生成
// ============================================================================
void GgmlBackend::generateAsync(const GenerateRequest& request, TokenCallback callback) {
    if (!isLoaded()) {
        TokenResult tr{};
        tr.is_eos = true;
        callback(tr);
        return;
    }

    state_ = InferenceState::DECODING;
    std::lock_guard<std::mutex> lock(impl_->mtx);
    impl_->cancelled = false;

    // v0.4：temp≤0 → 逐调用贪心链（即用即释），否则共享常驻链（v0.3 行为）
    llama_sampler* greedy = request.temperature <= 0.0f ? impl_->buildSampler(request.temperature) : nullptr;
    llama_sampler* sampler = greedy ? greedy : impl_->sampler;

    double gen_start = Impl::now_ms();
    std::vector<double> token_times;
    GenerateResult run{};

    // v0.4 §1.1：续写轮不清 KV、不加 BOS，仅提交胶水后的增量
    const bool cont = request.continue_session && impl_->n_past > 0;
    if (request.continue_session && !cont)
        EP_LOGW("continue_session=true 但无会话态——回退全新装载（P-1）");
    std::vector<llama_token> prompt_tokens = cont
        ? impl_->tokenize(impl_->chatGlue(request.prompt), false)
        : impl_->tokenize(request.prompt, true);
    if (prompt_tokens.empty()) {
        if (greedy) llama_sampler_free(greedy);
        run.generated_text = kPrefillError;
        run.total_time_ms = static_cast<float>(Impl::now_ms() - gen_start);
        compute_itl_metrics(run, token_times);
        impl_->last_metrics = run;  // 失败早退也定格本次(全零)指标，防 getLastMetrics 吐陈旧数据
        TokenResult tr{}; tr.is_eos = true;
        callback(tr);
        state_ = InferenceState::IDLE;
        return;
    }

    if (!cont) impl_->resetKVCache();     // 非续写路径逐字等价（原无条件 reset）
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(),
                                            static_cast<int32_t>(prompt_tokens.size()));
    if (llama_decode(impl_->ctx, batch) != 0) {
        if (greedy) llama_sampler_free(greedy);
        run.generated_text = kPrefillError;
        run.total_time_ms = static_cast<float>(Impl::now_ms() - gen_start);
        compute_itl_metrics(run, token_times);
        impl_->last_metrics = run;  // 同上：prompt 超长致 prefill 失败时，onDone 报本次全零而非上次结果
        TokenResult tr{}; tr.is_eos = true;
        callback(tr);
        state_ = InferenceState::IDLE;
        return;
    }
    impl_->n_past += static_cast<int32_t>(prompt_tokens.size());
    impl_->session_tokens.insert(impl_->session_tokens.end(),
                                 prompt_tokens.begin(), prompt_tokens.end());

    int32_t n_predict = request.max_new_tokens > 0 ? request.max_new_tokens : 256;
    llama_token new_token_id = 0;

    for (int32_t i = 0; i < n_predict; ++i) {
        if (impl_->cancelled) {
            TokenResult tr{}; tr.is_eos = true;
            callback(tr);
            break;
        }

        new_token_id = llama_sampler_sample(sampler, impl_->ctx, -1);
        if (llama_vocab_is_eog(impl_->vocab, new_token_id)) {
            TokenResult tr{}; tr.is_eos = true;
            callback(tr);
            break;
        }

        TokenResult tr{};
        tr.token_id = static_cast<int>(new_token_id);
        tr.token_text = impl_->tokenToPiece(new_token_id);
        tr.is_eos = false;

        token_times.push_back(Impl::now_ms());
        run.generated_text += tr.token_text;
        run.generated_tokens.push_back(tr.token_id);
        impl_->session_tokens.push_back(new_token_id);   // v0.4 §2.1：生成 token 入会话历史
        run.total_tokens++;
        if (run.total_tokens == 1)
            run.ttft_ms = static_cast<float>(Impl::now_ms() - gen_start);

        callback(tr);

        llama_batch next_batch = llama_batch_get_one(&new_token_id, 1);
        if (llama_decode(impl_->ctx, next_batch) != 0) {
            TokenResult end_tr{}; end_tr.is_eos = true;
            callback(end_tr);
            break;
        }
        impl_->n_past++;
    }

    run.total_time_ms = static_cast<float>(Impl::now_ms() - gen_start);
    if (run.total_tokens > 0 && run.total_time_ms > 0)
        run.tokens_per_sec = run.total_tokens / (run.total_time_ms / 1000.0f);
    compute_itl_metrics(run, token_times);
    impl_->last_metrics = run;      // 全程持锁，安全
    impl_->total_tokens_generated += run.total_tokens;
    impl_->total_generation_ms += run.total_time_ms;

    if (greedy) llama_sampler_free(greedy);
    state_ = InferenceState::IDLE;
}

void GgmlBackend::cancel() {
    impl_->cancelled = true;
}

// ============================================================================
// prefill
// ============================================================================
float GgmlBackend::prefill(const std::vector<int>& tokens) {
    if (!isLoaded() || tokens.empty()) return -1.0f;

    auto t0 = Impl::now_ms();
    impl_->resetKVCache();

    std::vector<llama_token> ltokens(tokens.begin(), tokens.end());
    llama_batch batch = llama_batch_get_one(ltokens.data(),
                                            static_cast<int32_t>(ltokens.size()));
    if (llama_decode(impl_->ctx, batch) != 0) return -1.0f;
    impl_->n_past += static_cast<int32_t>(tokens.size());

    return static_cast<float>(Impl::now_ms() - t0);
}

// ============================================================================
// decodeNext
// ============================================================================
TokenResult GgmlBackend::decodeNext() {
    TokenResult tr{};
    if (!isLoaded()) return tr;

    llama_token id = llama_sampler_sample(impl_->sampler, impl_->ctx, -1);

    tr.token_id = static_cast<int>(id);
    tr.token_text = impl_->tokenToPiece(id);
    tr.is_eos = llama_vocab_is_eog(impl_->vocab, id);

    float* logits = llama_get_logits(impl_->ctx);
    if (logits) {
        int n_vocab = llama_vocab_n_tokens(impl_->vocab);
        float max_logit = -1e9f;
        for (int i = 0; i < n_vocab; ++i)
            if (logits[i] > max_logit) max_logit = logits[i];
        float sum_exp = 0.0f;
        for (int i = 0; i < n_vocab; ++i)
            sum_exp += std::exp(logits[i] - max_logit);
        tr.probability = std::exp(logits[id] - max_logit) / sum_exp;
    }

    llama_batch batch = llama_batch_get_one(&id, 1);
    llama_decode(impl_->ctx, batch);
    impl_->n_past++;

    return tr;
}

// ============================================================================
// verifyTokens
// ============================================================================
std::vector<TokenResult> GgmlBackend::verifyTokens(const std::vector<int>& candidates) {
    std::vector<TokenResult> results;
    if (!isLoaded() || candidates.empty()) return results;

    float* logits = llama_get_logits(impl_->ctx);
    int n_vocab = llama_vocab_n_tokens(impl_->vocab);

    for (int token_id : candidates) {
        TokenResult tr{};
        tr.token_id = token_id;
        tr.token_text = impl_->tokenToPiece(static_cast<llama_token>(token_id));

        if (logits && token_id >= 0 && token_id < n_vocab) {
            float max_logit = -1e9f;
            for (int i = 0; i < n_vocab; ++i)
                if (logits[i] > max_logit) max_logit = logits[i];
            float sum_exp = 0.0f;
            for (int i = 0; i < n_vocab; ++i)
                sum_exp += std::exp(logits[i] - max_logit);
            tr.probability = std::exp(logits[token_id] - max_logit) / sum_exp;
        }

        llama_token lid = static_cast<llama_token>(token_id);
        llama_batch batch = llama_batch_get_one(&lid, 1);
        llama_decode(impl_->ctx, batch);
        impl_->n_past++;

        results.push_back(tr);
    }
    return results;
}

// ============================================================================
// 信息查询
// ============================================================================
HardwareInfo GgmlBackend::getHardwareInfo() const {
    HardwareInfo info{};
    info.soc_name = "llama.cpp";

#ifdef __ANDROID__
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        info.total_memory_bytes = si.totalram * si.mem_unit;
        info.available_memory_bytes = si.freeram * si.mem_unit;
    }
#else
    info.total_memory_bytes = 8ULL * 1024 * 1024 * 1024;
    info.available_memory_bytes = 4ULL * 1024 * 1024 * 1024;
#endif

    if (impl_->model) {
        char desc[256];
        llama_model_desc(impl_->model, desc, sizeof(desc));
        info.soc_name = std::string(desc);
    }
    info.gpu_name = "N/A";
    return info;
}

KVCacheInfo GgmlBackend::getKVCacheInfo() const {
    KVCacheInfo info{};
    if (impl_->ctx && impl_->model) {
        info.max_seq_len = static_cast<int>(llama_n_ctx(impl_->ctx));
        info.num_layers  = static_cast<int>(llama_model_n_layer(impl_->model));
        const int n_head     = static_cast<int>(llama_model_n_head(impl_->model));
        const int n_head_kv  = static_cast<int>(llama_model_n_head_kv(impl_->model));
        const int n_embd     = static_cast<int>(llama_model_n_embd(impl_->model));
        info.num_heads = n_head;
        info.head_dim  = n_head > 0 ? n_embd / n_head : 0;
        // v0.4 §4：按当前 ctx 实际量化类型计字节（ggml_row_size 与 llama 内部分配同式，不猜）
        ggml_type tk, tv; kv_types(impl_->config.kv_cache_bits, tk, tv);
        const size_t nkv = static_cast<size_t>(n_head_kv > 0 ? n_head_kv : n_head);
        const size_t per_tok = static_cast<size_t>(info.num_layers) * nkv
            * (ggml_row_size(tk, info.head_dim) + ggml_row_size(tv, info.head_dim));
        info.used_memory_bytes  = static_cast<size_t>(impl_->n_past) * per_tok;
        info.total_memory_bytes = static_cast<size_t>(info.max_seq_len) * per_tok;
    }
    return info;
}

InferenceState GgmlBackend::getState() const { return state_; }

std::string GgmlBackend::getName() const { return "llama.cpp (GGML)"; }

GenerateResult GgmlBackend::getLastMetrics() const {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    return impl_->last_metrics;
}

std::vector<uint8_t> GgmlBackend::exportKVCache() const {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    if (!isLoaded()) return {};
    size_t sz = llama_state_seq_get_size(impl_->ctx, 0);
    if (sz == 0) return {};
    std::vector<uint8_t> buf(sz);
    size_t got = llama_state_seq_get_data(impl_->ctx, buf.data(), sz, 0);
    if (got == 0) return {};
    buf.resize(got);
    return buf;
}
bool GgmlBackend::importKVCache(const std::vector<uint8_t>& data) {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    if (!isLoaded() || data.empty()) return false;
    if (llama_state_seq_set_data(impl_->ctx, data.data(), data.size(), 0) == 0) return false;
    // 缓冲版无 token 历史载体：长度未知 → n_past=0、session_tokens 清空（文件版恢复不走此路）
    impl_->n_past = 0; impl_->session_tokens.clear();
    EP_LOGW("importKVCache(缓冲): 无 token 历史，会话导出在此路径后不可用");
    return true;
}

bool GgmlBackend::saveSessionFile(const std::string& base) {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    if (!isLoaded()) { EP_LOGE("saveSessionFile: 引擎未加载"); return false; }
    if (impl_->session_tokens.empty()) { EP_LOGE("saveSessionFile: 无会话 token 历史（未跑过生成）"); return false; }
    const std::string kv = base + ".kvdat", meta = kv + ".json";
    size_t n = llama_state_seq_save_file(impl_->ctx, kv.c_str(), 0,
                                         impl_->session_tokens.data(), impl_->session_tokens.size());
    if (n == 0) { EP_LOGE("saveSessionFile: llama_state_seq_save_file 失败 → %s", kv.c_str()); return false; }
    struct stat st{}; stat(impl_->config.model_path.c_str(), &st);
    FILE* f = fopen(meta.c_str(), "wb");
    if (!f) { EP_LOGE("saveSessionFile: sidecar 不可写 %s", meta.c_str()); return false; }
    fprintf(f, "{\"schema\":\"epkv1\",\"model_path\":\"%s\",\"model_size_bytes\":%lld,\"n_ctx\":%d,\"kv_bits\":%d,\"n_tokens\":%zu,\"saved_at_ms\":%lld}",
            json_escape_basic(impl_->config.model_path).c_str(), (long long)st.st_size,
            (int)llama_n_ctx(impl_->ctx), impl_->config.kv_cache_bits,
            impl_->session_tokens.size(), wall_ms());
    fclose(f);
    EP_LOGI("saveSessionFile: %zu tokens → %s", impl_->session_tokens.size(), kv.c_str());
    return true;
}

bool GgmlBackend::loadSessionFile(const std::string& base) {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    if (!isLoaded()) { EP_LOGE("loadSessionFile: 引擎未加载"); return false; }
    const std::string kv = base + ".kvdat", meta = kv + ".json";
    auto reject = [&](const char* why) { EP_LOGE("loadSessionFile 拒绝: %s (%s)", why, base.c_str()); return false; };
    // 1 两文件存在
    struct stat st{};
    if (stat(kv.c_str(), &st) != 0 || st.st_size <= 0) return reject("kvdat 缺失");
    FILE* mf = fopen(meta.c_str(), "rb"); if (!mf) return reject("sidecar 缺失");
    std::string j(65536, '\0'); size_t rn = fread(&j[0], 1, j.size(), mf); fclose(mf); j.resize(rn);
    // 2 schema
    std::string v;
    if (!sidecar_fetch(j, "schema", v) || v != "epkv1") return reject("schema 非 epkv1");
    // 3 模型指纹：路径 + 字节数
    std::string mpath; long long msz = -1;
    if (!sidecar_fetch(j, "model_path", mpath) || mpath != impl_->config.model_path) return reject("model_path 不符");
    { std::string t; if (!sidecar_fetch(j, "model_size_bytes", t)) return reject("size 缺失"); msz = atoll(t.c_str()); }
    if (stat(impl_->config.model_path.c_str(), &st) != 0 || (long long)st.st_size != msz) return reject("模型文件已变更");
    // 4 kv_bits
    { std::string t; int kb = 16; if (sidecar_fetch(j, "kv_bits", t)) kb = atoi(t.c_str());
      if (kb != impl_->config.kv_cache_bits) return reject("kv_bits 与当前上下文不符"); }
    // 5 载入（sidecar n_tokens 作容量；llama 原生校验兜底）
    size_t cap = 0; { std::string t; if (sidecar_fetch(j, "n_tokens", t)) cap = (size_t)strtoull(t.c_str(), nullptr, 10); }
    if (cap == 0 || cap > (size_t)llama_n_ctx(impl_->ctx)) return reject("n_tokens 非法/超上下文");
    impl_->resetKVCache();                       // 校验全过后才清场（旧会话在拒绝路径完好）
    std::vector<llama_token> toks(cap);
    size_t got = 0;
    if (llama_state_seq_load_file(impl_->ctx, kv.c_str(), 0, toks.data(), cap, &got) == 0)
        return reject("llama 状态装载失败（版本/长度/容量）");
    toks.resize(got);
    impl_->session_tokens = std::move(toks);
    impl_->n_past = (int32_t)got;
    EP_LOGI("loadSessionFile: %zu tokens 恢复（可 continueSession 续写）", got);
    return true;
}

void GgmlBackend::clearKVCache() { impl_->resetKVCache(); }

bool GgmlBackend::compressKVCache(int target_bits) {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    if (!isLoaded()) { EP_LOGE("compressKVCache: 未加载"); return false; }
    if (target_bits != 16 && target_bits != 8 && target_bits != 4) { EP_LOGE("compressKVCache: 非法档位 %d", target_bits); return false; }
    auto cp = make_cparams(impl_->config);
    cp.n_ctx = llama_n_ctx(impl_->ctx);                   // 保当前 ctx，预算降档不重演
    // T3 裁定②：不覆写 n_threads——make_cparams 自 config 解析同值，直抄会把 auto(0) 打成 0 线程
    ggml_type tk, tv; kv_types(target_bits, tk, tv);
    cp.type_k = tk; cp.type_v = tv;
    llama_free(impl_->ctx); impl_->ctx = nullptr;
    impl_->config.kv_cache_bits = target_bits;
    impl_->ctx = llama_init_from_model(impl_->model, cp);
    if (!impl_->ctx) { EP_LOGE("compressKVCache: 重建失败（KV 已弃，模型仍可用重载恢复）"); state_ = InferenceState::ERROR; return false; }
    impl_->n_past = 0; impl_->session_tokens.clear();     // 重建=清零（spec 裁定④：诚实"需重灌历史"）
    EP_LOGI("compressKVCache: ctx 已按 bits=%d 重建，KV 清零，历史需重新 prefill", target_bits);
    return true;
}

} // namespace edgepilot
