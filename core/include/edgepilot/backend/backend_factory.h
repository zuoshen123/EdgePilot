#pragma once

#include "edgepilot/backend/inference_backend.h"
#include "edgepilot/common/types.h"
#include <memory>

namespace edgepilot {

/**
 * 后端工厂
 *
 * 根据设备硬件能力自动选择最优推理后端。
 * 选择策略:
 *   1. 检测 NPU 可用性 → ExecuTorch (QNN delegate)
 *   2. 检测 GPU 可用性 → ncnn (Vulkan) / GGML (Metal)
 *   3. 回退到 CPU → ExecuTorch (XNNPACK) / GGML
 */
class BackendFactory {
public:
    /// 创建最优后端（自动检测硬件）
    static std::unique_ptr<InferenceBackend> createOptimal();

    /// 创建指定类型的后端
    static std::unique_ptr<InferenceBackend> create(BackendType type);

    /// 创建指定设备的后端
    static std::unique_ptr<InferenceBackend> createForDevice(ComputeDevice device);

    /// 检测当前设备的硬件能力
    static HardwareInfo detectHardware();

    /// 列出所有可用的后端
    static std::vector<BackendType> listAvailable();

    /// 获取推荐的模型配置（基于硬件能力）
    static ModelConfig getRecommendedConfig(const HardwareInfo& hw_info);
};

} // namespace edgepilot
