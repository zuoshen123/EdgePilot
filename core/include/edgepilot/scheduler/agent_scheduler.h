#pragma once

#include "edgepilot/common/types.h"
#include "edgepilot/backend/inference_backend.h"
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <functional>

namespace edgepilot {

using DegradationLevel = int;  // 0-4

/**
 * 多 Agent 并发调度器
 *
 * 解决多个 Agent 同时请求推理时的资源竞争问题。
 *
 * 调度策略:
 *   1. 优先级队列: 高优先级 Agent 优先获得推理资源
 *   2. 热感知: 温度接近阈值时自动降级 (减小 batch / 降精度)
 *   3. 内存感知: 预估内存峰值，防止 OOM
 *   4. 时间片: 低优先级 Agent 在高优先级空闲时获得推理机会
 *   5. 抢占: REALTIME 优先级可抢占正在执行的低优先级任务
 *
 * 降级策略 (从轻到重):
 *   Level 0: 无降级
 *   Level 1: 减小 batch size
 *   Level 2: KV Cache INT8 → INT4
 *   Level 3: 切换到更小的模型
 *   Level 4: 暂停低优先级 Agent
 */
class AgentScheduler {
public:
    using InferenceFunc = std::function<GenerateResult(const GenerateRequest&)>;

    struct Config {
        int max_concurrent_inference;     // 最大并发推理数
        float thermal_warning_threshold;  // 温度警告阈值 (0-1)
        float thermal_critical_threshold; // 温度危险阈值
        float memory_warning_threshold;   // 内存警告阈值
        int scheduling_interval_ms;       // 调度间隔
        bool enable_preemption;           // 是否允许抢占
    };

    AgentScheduler();
    ~AgentScheduler();

    /// 初始化调度器
    Status initialize(const Config& config, InferenceBackend* backend);

    /// 提交推理请求 (非阻塞)
    int submitRequest(const AgentRequest& request);

    /// 等待请求完成
    GenerateResult waitForResult(int request_id,
                                 std::chrono::milliseconds timeout);

    /// 取消请求
    void cancelRequest(int request_id);

    /// 查询请求状态
    InferenceState queryState(int request_id);

    // ---- 资源状态 ----

    struct ResourceStatus {
        HardwareInfo hardware;
        int pending_requests;
        int active_inferences;
        DegradationLevel current_degradation;
        float thermal_pressure;       // 0-1
        float memory_pressure;        // 0-1
        std::vector<int> queued_agents;
        std::vector<int> active_agents;
    };
    ResourceStatus getResourceStatus() const;

    /// 强制设置降级级别 (用于测试)
    void forceDegradationLevel(DegradationLevel level);

    // ---- 回调 ----

    using StatusCallback = std::function<void(int request_id,
                                               InferenceState state,
                                               const GenerateResult& result)>;
    void setStatusCallback(StatusCallback callback);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * 功耗管理器
 *
 * 监控设备温度和电池状态，在必要时触发降级策略。
 */
class PowerManager {
public:
    struct ThermalPolicy {
        float temp_warning;     // 警告温度 (°C)
        float temp_critical;    // 危险温度
        float temp_shutdown;    // 强制停止温度
        int cooldown_ms;        // 降频后冷却时间
    };

    struct PowerSnapshot {
        float battery_level;
        float current_ma;       // 放电电流
        float power_mw;         // 实时功耗
        float cpu_temp;
        float gpu_temp;
        float battery_temp;
        int thermal_status;     // 系统热状态
    };

    /// 设置热策略
    static void setThermalPolicy(const ThermalPolicy& policy);

    /// 获取当前功耗快照
    static PowerSnapshot getCurrentPower();

    /// 判断是否应该降级
    static DegradationLevel recommendDegradation();

    /// 计算每 token 能耗 (mJ)
    static float energyPerToken(float power_mw, float tokens_per_sec);
};

} // namespace edgepilot
