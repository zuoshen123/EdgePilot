#include "edgepilot/platform/platform_sampler.h"

#include "edgepilot/common/log.h"

#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef __ANDROID__
#include <dirent.h>
#endif

namespace edgepilot {

const char* samplerStatusToString(SamplerStatus s) {
    switch (s) {
        case SamplerStatus::OK:          return "OK";
        case SamplerStatus::PERM_DENIED: return "PERM_DENIED";
        case SamplerStatus::ABSENT:      break;
    }
    return "ABSENT";
}

double sampler_now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

#ifdef __ANDROID__

namespace {

// 读文件首行首整数；EACCES 置 *perm_denied
bool read_long_file(const std::string& path, long long* out, bool* perm_denied) {
    FILE* f = std::fopen(path.c_str(), "re");
    if (!f) {
        if (perm_denied && errno == EACCES) *perm_denied = true;
        return false;
    }
    char buf[64] = {0};
    bool got = std::fgets(buf, sizeof(buf), f) != nullptr;
    std::fclose(f);
    if (!got) return false;
    char* end = nullptr;
    long long v = std::strtoll(buf, &end, 10);
    if (end == buf) return false;
    *out = v;
    return true;
}

// 读文件首行（去行尾 \r\n）；EACCES 置 *perm_denied
std::string read_line_file(const std::string& path, bool* perm_denied) {
    FILE* f = std::fopen(path.c_str(), "re");
    if (!f) {
        if (perm_denied && errno == EACCES) *perm_denied = true;
        return "";
    }
    char buf[128] = {0};
    bool got = std::fgets(buf, sizeof(buf), f) != nullptr;
    std::fclose(f);
    std::string s = got ? buf : "";
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    return s;
}

// thermal_zone type 是否算 CPU/GPU 计算簇（spec §①，大小写不敏感）
bool zone_type_is_compute(const std::string& t) {
    std::string s;
    s.reserve(t.size());
    for (char c : t) s += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s.find("cpu") != std::string::npos ||
           s.find("gpu") != std::string::npos ||
           s.find("cluster") != std::string::npos ||
           s.find("llama") != std::string::npos;
}

} // namespace

bool PlatformSampler::probe() {
    cap_ = Capability{};
    zone_paths_.clear();
    bool saw_perm = false;

    // ---- 功耗：枚举 /sys/class/power_supply，找 type==Battery，优先目录名 battery ----
    std::string found;
    if (DIR* d = ::opendir("/sys/class/power_supply")) {
        for (dirent* e = ::readdir(d); e != nullptr; e = ::readdir(d)) {
            std::string name = e->d_name;
            if (name == "." || name == "..") continue;
            bool perm = false;
            std::string type = read_line_file("/sys/class/power_supply/" + name + "/type", &perm);
            if (perm) saw_perm = true;
            if (type == "Battery" && (found.empty() || name == "battery")) {
                found = name;
                if (name == "battery") break;
            }
        }
        ::closedir(d);
    } else {
        if (errno == EACCES) saw_perm = true;
    }
    if (!found.empty()) {
        cap_.power_path = "/sys/class/power_supply/" + found;
        bool perm = false;
        long long cur = 0;
        if (read_long_file(cap_.power_path + "/current_now", &cur, &perm)) {
            cap_.power = SamplerStatus::OK;
        } else {
            cap_.power = perm ? SamplerStatus::PERM_DENIED : SamplerStatus::ABSENT;
        }
    } else {
        cap_.power = saw_perm ? SamplerStatus::PERM_DENIED : SamplerStatus::ABSENT;
    }

    // ---- 温度：枚举 thermal_zone*，命中 cpu/gpu 类 ----
    saw_perm = false;
    if (DIR* td = ::opendir("/sys/class/thermal")) {
        for (dirent* e = ::readdir(td); e != nullptr; e = ::readdir(td)) {
            std::string name = e->d_name;
            if (name.rfind("thermal_zone", 0) != 0) continue;
            bool perm = false;
            std::string type = read_line_file("/sys/class/thermal/" + name + "/type", &perm);
            if (perm) saw_perm = true;
            if (zone_type_is_compute(type)) {
                cap_.cpu_zone_types.push_back(type);
                zone_paths_.push_back("/sys/class/thermal/" + name + "/temp");
            }
        }
        ::closedir(td);
        cap_.thermal = !zone_paths_.empty() ? SamplerStatus::OK
                     : (saw_perm ? SamplerStatus::PERM_DENIED : SamplerStatus::ABSENT);
    } else {
        cap_.thermal = (errno == EACCES) ? SamplerStatus::PERM_DENIED : SamplerStatus::ABSENT;
    }

    // ---- 内存（自身进程，Android 上恒可读）----
    if (FILE* f = std::fopen("/proc/self/status", "re")) {
        std::fclose(f);
        cap_.mem = SamplerStatus::OK;
    } else {
        cap_.mem = (errno == EACCES) ? SamplerStatus::PERM_DENIED : SamplerStatus::ABSENT;
    }

    // ---- CPU（/proc/stat；本版仅探能力，不进导出列）----
    if (FILE* f = std::fopen("/proc/stat", "re")) {
        std::fclose(f);
        cap_.cpu = SamplerStatus::OK;
    } else {
        cap_.cpu = (errno == EACCES) ? SamplerStatus::PERM_DENIED : SamplerStatus::ABSENT;
    }

    EP_LOGI("sampler probe: power=%s(%s) thermal=%s(%zu zones) mem=%s cpu=%s",
            samplerStatusToString(cap_.power), cap_.power_path.c_str(),
            samplerStatusToString(cap_.thermal), cap_.cpu_zone_types.size(),
            samplerStatusToString(cap_.mem), samplerStatusToString(cap_.cpu));

    return cap_.power == SamplerStatus::OK || cap_.thermal == SamplerStatus::OK ||
           cap_.mem == SamplerStatus::OK || cap_.cpu == SamplerStatus::OK;
}

bool PlatformSampler::readPower(PowerSample& out) {
    if (cap_.power != SamplerStatus::OK) return false;
    bool perm = false;
    long long cur_ua = 0, vol_uv = 0;
    if (!read_long_file(cap_.power_path + "/current_now", &cur_ua, &perm) ||
        !read_long_file(cap_.power_path + "/voltage_now", &vol_uv, &perm)) {
        if (perm) cap_.power = SamplerStatus::PERM_DENIED;
        return false;
    }
    if (vol_uv <= 0 || vol_uv > 20000000LL) return false;  // 电压异常：样本作废（spec §①）
    out.t_ms = sampler_now_ms();
    out.power_mw = std::fabs(static_cast<double>(cur_ua)) *
                   static_cast<double>(vol_uv) / 1.0e9;    // µA×µV/1e9 = mW
    return true;
}

bool PlatformSampler::readThermal(ThermalSample& out) {
    if (cap_.thermal != SamplerStatus::OK) return false;
    bool any_perm = false;
    bool got = false;
    long long best = 0;
    for (const std::string& p : zone_paths_) {
        bool perm = false;
        long long t = 0;
        if (read_long_file(p, &t, &perm)) {
            if (perm) any_perm = true;
            if (t > 0 && t < 150000) {          // 毫度合理域 0–150℃，越界弃值
                if (!got || t > best) { best = t; got = true; }
            }
        } else if (perm) {
            any_perm = true;
        }
    }
    if (!got) {
        if (any_perm) cap_.thermal = SamplerStatus::PERM_DENIED;
        return false;
    }
    out.t_ms = sampler_now_ms();
    out.cpu_max_c = static_cast<float>(best) / 1000.0f;
    return true;
}

bool PlatformSampler::readMem(MemSample& out) {
    if (cap_.mem != SamplerStatus::OK) return false;
    out = MemSample{};

    FILE* f = std::fopen("/proc/self/status", "re");
    if (!f) { cap_.mem = SamplerStatus::PERM_DENIED; return false; }
    char line[256];
    while (std::fgets(line, sizeof(line), f)) {
        if (std::strncmp(line, "VmRSS:", 6) == 0) {
            out.vmrss_kb = std::strtoull(line + 6, nullptr, 10);
            break;
        }
    }
    std::fclose(f);

    // smaps_rollup 缺（旧内核）→ pss 保持 0；峰值取 max，0 不会误抬
    f = std::fopen("/proc/self/smaps_rollup", "re");
    if (f) {
        while (std::fgets(line, sizeof(line), f)) {
            if (std::strncmp(line, "Pss:", 4) == 0) {
                out.pss_kb = std::strtoull(line + 4, nullptr, 10);
                break;
            }
        }
        std::fclose(f);
    }
    out.t_ms = sampler_now_ms();
    return out.vmrss_kb > 0 || out.pss_kb > 0;
}

#else  // !__ANDROID__：桌面调试宿主全降级（spec §①，core 不得有 Android 硬依赖）

bool PlatformSampler::probe() {
    cap_ = Capability{};
    zone_paths_.clear();
    return false;
}
bool PlatformSampler::readPower(PowerSample&)   { return false; }
bool PlatformSampler::readThermal(ThermalSample&) { return false; }
bool PlatformSampler::readMem(MemSample&)       { return false; }

#endif

} // namespace edgepilot
