#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace edgepilot {

/// 采集通道能力状态（spec §①：不伪造 —— 不可用通道不产生样本）
enum class SamplerStatus { OK, PERM_DENIED, ABSENT };

const char* samplerStatusToString(SamplerStatus s);

/// steady_clock 毫秒（与 GgmlBackend::Impl::now_ms 同时钟域，可与 token_times 对齐）
double sampler_now_ms();

/// probe() 结果，probe 后只读
struct Capability {
    SamplerStatus power = SamplerStatus::ABSENT;
    SamplerStatus thermal = SamplerStatus::ABSENT;
    SamplerStatus mem = SamplerStatus::ABSENT;
    SamplerStatus cpu = SamplerStatus::ABSENT;
    std::string power_path;                      // 命中的 supply 目录（含 type==Battery）
    std::vector<std::string> cpu_zone_types;     // 命中 cpu/gpu 类的 thermal_zone type
};

struct PowerSample   { double t_ms = 0; double power_mw = 0; };
struct ThermalSample { double t_ms = 0; float cpu_max_c = 0; };
struct MemSample     { double t_ms = 0; size_t pss_kb = 0; size_t vmrss_kb = 0; };

/// sysfs/procfs 平台采样器。非 Android 宿主 probe() 全 ABSENT、read* 恒 false。
class PlatformSampler {
public:
    bool probe();                                // 可重复调用（重置能力矩阵与 zone 缓存）；任一通道 OK 返回 true
    const Capability& capability() const { return cap_; }

    bool readPower(PowerSample& out);            // |current_µA| × voltage_µV / 1e9 → mW
    bool readThermal(ThermalSample& out);        // 命中 zone 的 temp 最大值（毫度→℃）
    bool readMem(MemSample& out);                // smaps_rollup Pss + status VmRSS（KB）

private:
    Capability cap_;
    std::vector<std::string> zone_paths_;        // probe 命中的 <zone>/temp 路径
};

} // namespace edgepilot
