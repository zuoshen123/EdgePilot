#pragma once

#include "edgepilot/common/types.h"
#include <string>
#include <vector>
#include <functional>
#include <thread>
#include <mutex>
#include <atomic>

namespace edgepilot {

/**
 * 全链路性能指标采集器
 *
 * 采集维度:
 *   1. 延迟: TTFT, ITL (avg/p95/p99), E2E
 *   2. 吞吐: tokens/sec, prefill tokens/sec
 *   3. 资源: CPU/GPU/NPU 使用率, 内存峰值
 *   4. 功耗: 实时功耗, 每 token 能耗, 温度
 *   5. 质量: 投机采样接受率, 缓存命中率
 *
 * 数据流向:
 *   采集器 → SQLite 持久化
 *          → 实时回调 (UI 展示)
 *          → 离线分析 (Python 脚本)
 */
class MetricsCollector {
public:
    struct Config {
        int sample_interval_ms;        // 采样间隔 (默认 100ms)
        bool enable_sqlite;            // 是否写入 SQLite
        std::string db_path;           // SQLite 文件路径
        bool enable_realtime_callback; // 是否启用实时回调
        int max_history_size;          // 内存中保留的最大历史条数
    };

    // ITL 百分位数计算
    struct ITLStats {
        float avg_ms;
        float p50_ms;
        float p95_ms;
        float p99_ms;
        float min_ms;
        float max_ms;
        int sample_count;
    };

    // 单次推理的完整指标
    struct InferenceRecord {
        int request_id;
        int agent_id;
        std::chrono::steady_clock::time_point start_time;
        std::chrono::steady_clock::time_point end_time;

        // 延迟
        float ttft_ms;
        std::vector<float> itl_samples_ms;  // 每个 token 的延迟
        ITLStats itl_stats;
        float e2e_latency_ms;

        // 吞吐
        float tokens_per_sec;
        float prefill_tokens_per_sec;
        int total_tokens;

        // 资源快照 (推理期间的峰值)
        float cpu_peak_pct;
        float gpu_peak_pct;
        size_t memory_peak_mb;
        size_t kv_cache_peak_mb;

        // 功耗
        float power_avg_mw;
        float power_peak_mw;
        float thermal_max_celsius;
        int thermal_throttle_count;

        // 投机采样
        int draft_tokens_proposed;
        int draft_tokens_accepted;
        float acceptance_rate;

        // 缓存
        bool cache_hit;
        int cache_prefix_length;
    };

    MetricsCollector();
    ~MetricsCollector();

    /// 初始化采集器
    Status initialize(const Config& config);

    /// 启动后台采集线程
    void start();

    /// 停止采集
    void stop();

    /// 是否正在采集
    bool isRunning() const;

    // ---- 记录推理事件 ----

    /// 标记推理开始
    void beginInference(int request_id, int agent_id);

    /// 记录 TTFT
    void recordTTFT(int request_id, float ttft_ms);

    /// 记录一个 ITL 样本
    void recordITL(int request_id, float itl_ms);

    /// 标记推理结束
    void endInference(int request_id, const GenerateResult& result);

    // ---- 查询 ----

    /// 获取最近 N 次推理的统计
    PerformanceMetrics getRecentMetrics(int count = 10) const;

    /// 获取某次推理的详细记录
    InferenceRecord getInferenceRecord(int request_id) const;

    /// 获取所有历史记录
    std::vector<InferenceRecord> getHistory() const;

    /// 计算 ITL 统计
    static ITLStats calculateITLStats(const std::vector<float>& samples);

    // ---- 导出 ----

    /// 导出到 SQLite
    void exportToSQLite(const std::string& path) const;

    /// 导出为 JSON 字符串
    std::string exportToJSON() const;

    /// 导出为 CSV
    void exportToCSV(const std::string& path) const;

    // ---- 实时回调 ----

    using MetricsCallback = std::function<void(const PerformanceMetrics&)>;
    void setMetricsCallback(MetricsCallback callback);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * 性能 Profiler
 *
 * 用于定位推理瓶颈，生成可读的性能分析报告。
 */
class Profiler {
public:
    enum class Event {
        MODEL_LOAD,
        TOKENIZE,
        PREFILL,
        DECODE,
        DRAFT_INFERENCE,     // 投机采样: draft 模型
        TARGET_VERIFY,       // 投机采样: target 验证
        KV_CACHE_SERIALIZE,
        KV_CACHE_DESERIALIZE,
        KV_CACHE_COMPRESS,
        KV_CACHE_DECOMPRESS,
        CACHE_LOOKUP,
        CACHE_STORE,
        SCHEDULER_DISPATCH,
        MEMORY_ALLOCATION,
        DEVICE_SYNC
    };

    /// 记录事件开始
    static void beginEvent(Event event);

    /// 记录事件结束
    static void endEvent(Event event);

    /// 获取事件耗时统计
    struct EventStats {
        float total_ms;
        float avg_ms;
        float min_ms;
        float max_ms;
        int call_count;
        float pct_of_total;
    };
    static EventStats getEventStats(Event event);

    /// 生成火焰图数据 (JSON 格式，可用 Chrome Tracing 打开)
    static std::string exportChromeTrace();

    /// 生成可读报告
    static std::string generateReport();

    /// 重置所有统计
    static void reset();
};

} // namespace edgepilot
