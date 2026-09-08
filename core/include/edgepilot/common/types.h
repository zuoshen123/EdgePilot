#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <functional>
#include <chrono>
#include <memory>
#include <unordered_map>

namespace edgepilot {

// ============================================================
// 基础类型定义
// ============================================================

enum class BackendType {
    EXECUTORCH,
    NCNN,
    GGML,
    AUTO  // 自动检测最优后端
};

enum class ComputeDevice {
    CPU,
    GPU_VULKAN,
    GPU_METAL,
    NPU_QUALCOMM,
    NPU_APPLE_ANE,
    AUTO
};

enum class Priority : int {
    REALTIME  = 0,  // 用户正在交互
    HIGH      = 1,  // 前台 Agent
    NORMAL    = 2,  // 后台活跃 Agent
    LOW       = 3,  // 后台预加载
    IDLE      = 4   // 仅充电时执行
};

// 硬件能力描述
struct HardwareInfo {
    ComputeDevice device;
    bool has_npu;
    bool has_gpu;
    size_t total_memory_bytes;
    size_t available_memory_bytes;
    int compute_units;
    float thermal_state;       // 0.0 (正常) - 1.0 (过热)
    float battery_level;       // 0.0 - 1.0
    bool is_charging;
    std::string gpu_name;
    std::string soc_name;      // e.g. "Snapdragon 8 Gen3"
};

// 模型配置
struct ModelConfig {
    std::string model_path;    // GGUF 文件路径
    BackendType backend;
    ComputeDevice device;
    int context_length;
    int batch_size;
    int threads;               // CPU 线程数
    bool use_mmap;             // 内存映射加载
    int gpu_layers;            // 卸载到 GPU 的层数 (-1 = 全部)
    // 量化配置
    int kv_cache_bits;         // KV Cache 量化位数 (4, 8, 16)
    bool enable_flash_attn;
};

// 推理请求
struct GenerateRequest {
    std::vector<int> tokens;         // 输入 token ids
    std::string prompt;              // 原始 prompt (如果需要重新 tokenize)
    int max_new_tokens;
    float temperature;
    int top_k;
    float top_p;
    float repeat_penalty;
    std::vector<int> stop_tokens;
    // 调度信息
    int agent_id;
    Priority priority;
    std::chrono::milliseconds timeout;
};

// 逐 token 回调
struct TokenResult {
    int token_id;
    std::string token_text;
    float probability;
    bool is_eos;
};

using TokenCallback = std::function<void(const TokenResult&)>;

// 推理结果
struct GenerateResult {
    std::vector<int> generated_tokens;
    std::string generated_text;
    // 性能指标
    float ttft_ms;
    float itl_avg_ms;
    float itl_p99_ms;
    float tokens_per_sec;
    int total_tokens;
    float total_time_ms;
    // 投机采样统计
    int draft_tokens_proposed;
    int draft_tokens_accepted;
    float acceptance_rate;
};

// KV Cache 布局信息
struct KVCacheInfo {
    int num_layers;
    int num_heads;
    int head_dim;
    int max_seq_len;
    size_t total_memory_bytes;
    size_t used_memory_bytes;
};

// Tensor 信息
struct TensorInfo {
    std::vector<int64_t> shape;
    enum DType { FP32, FP16, INT8, INT4 } dtype;
    size_t nbytes;
};

// 性能指标
struct PerformanceMetrics {
    // 延迟
    float ttft_ms;
    float itl_avg_ms;
    float itl_p95_ms;
    float itl_p99_ms;
    float e2e_latency_ms;

    // 吞吐
    float tokens_per_sec;
    float prefill_tokens_per_sec;

    // 资源
    float cpu_usage_pct;
    float gpu_usage_pct;
    float npu_usage_pct;
    size_t memory_peak_mb;
    size_t kv_cache_size_mb;

    // 功耗
    float power_mw;
    float energy_per_token_mj;
    float thermal_celsius;
    int thermal_throttle_events;

    // 推理质量
    float draft_acceptance_rate;
    float cache_hit_rate;

    std::chrono::steady_clock::time_point timestamp;
};

// Agent 请求 (用于调度器)
struct AgentRequest {
    int agent_id;
    std::string agent_name;
    Priority priority;
    GenerateRequest inference_request;
    size_t estimated_memory_bytes;
    std::chrono::milliseconds timeout;
    bool allow_degradation;  // 允许降级执行 (更小模型/更低精度)
};

// 调度结果
struct ScheduleDecision {
    int agent_id;
    bool accepted;           // false = 排队等待
    ComputeDevice assigned_device;
    int assigned_batch_size;
    bool degraded;           // 是否降级执行
    std::chrono::milliseconds estimated_wait;
    std::string reject_reason; // 拒绝原因 (如果 rejected)
};

// 推理状态
enum class InferenceState {
    IDLE,
    LOADING,
    PREFILL,
    DECODING,
    VERIFYING,   // 投机采样验证阶段
    COMPLETED,
    ERROR,
    CANCELLED
};

// 状态码
enum class Status {
    OK,
    MODEL_NOT_FOUND,
    MODEL_LOAD_FAILED,
    INVALID_CONFIG,
    OUT_OF_MEMORY,
    HARDWARE_ERROR,
    TIMEOUT,
    CANCELLED,
    NOT_SUPPORTED,
    UNKNOWN_ERROR
};

inline const char* statusToString(Status s) {
    switch (s) {
        case Status::OK:                return "OK";
        case Status::MODEL_NOT_FOUND:   return "MODEL_NOT_FOUND";
        case Status::MODEL_LOAD_FAILED: return "MODEL_LOAD_FAILED";
        case Status::INVALID_CONFIG:    return "INVALID_CONFIG";
        case Status::OUT_OF_MEMORY:     return "OUT_OF_MEMORY";
        case Status::HARDWARE_ERROR:    return "HARDWARE_ERROR";
        case Status::TIMEOUT:           return "TIMEOUT";
        case Status::CANCELLED:         return "CANCELLED";
        case Status::NOT_SUPPORTED:     return "NOT_SUPPORTED";
        case Status::UNKNOWN_ERROR:     return "UNKNOWN_ERROR";
    }
    return "UNKNOWN";
}

} // namespace edgepilot
