#pragma once

#include "edgepilot/common/types.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <list>

namespace edgepilot {

/**
 * Prompt Cache 管理器
 *
 * 基于 token 前缀匹配的缓存系统。
 * 当新的请求与已缓存的请求共享前缀时，直接复用 KV Cache，
 * 跳过 prefill 阶段，将 TTFT 从数百毫秒降到接近 0。
 *
 * 典型场景:
 *   - 系统 prompt (通常 200-500 tokens) 每次请求都相同
 *   - 多轮对话中，历史消息的 KV Cache 可以复用
 *   - Agent 的固定指令前缀
 */
class PromptCacheManager {
public:
    struct Config {
        size_t max_memory_bytes;       // 缓存池最大内存
        int max_entries;               // 最大缓存条目数
        float eviction_threshold;      // 内存使用超过此比例时开始淘汰
        bool enable_disk_cache;        // 是否持久化到磁盘
        std::string cache_dir;         // 磁盘缓存目录
    };

    struct CacheEntry {
        std::string entry_id;
        std::vector<int> token_prefix;
        std::vector<uint8_t> kv_data;  // 序列化的 KV Cache
        int prefix_length;             // 缓存的 token 数
        size_t memory_bytes;
        std::chrono::steady_clock::time_point created_at;
        std::chrono::steady_clock::time_point last_used;
        int use_count;
        int agent_id;                  // 归属的 Agent
    };

    struct CacheHit {
        bool found;
        std::string entry_id;
        int matched_length;            // 匹配的前缀长度
        int total_length;              // 请求的总长度
        float match_ratio;             // matched / total
        const CacheEntry* entry;       // 命中的缓存条目
    };

    PromptCacheManager();
    ~PromptCacheManager();

    /// 初始化缓存管理器
    Status initialize(const Config& config);

    /// 查找最长前缀匹配
    CacheHit findBestCache(const std::vector<int>& tokens, int agent_id = -1);

    /// 存入缓存
    Status store(const std::vector<int>& tokens,
                 const std::vector<uint8_t>& kv_data,
                 int agent_id = -1);

    /// 使缓存失效
    void invalidate(const std::string& entry_id);

    /// 清除某个 Agent 的所有缓存
    void invalidateForAgent(int agent_id);

    /// 清除所有缓存
    void clear();

    // ---- 统计 ----

    struct CacheStats {
        int total_entries;
        size_t total_memory_bytes;
        int total_lookups;
        int cache_hits;
        float hit_rate;
        float avg_match_ratio;
        float memory_usage_ratio;
    };
    CacheStats getStats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * KV Cache 压缩器
 *
 * 支持 FP16 → INT4/INT8 量化压缩，大幅减少内存占用。
 * INT4 量化对 attention scores 精度影响极小 (<0.5% perplexity 增加)。
 */
class KVCacheCompressor {
public:
    /// 压缩 KV Cache
    static std::vector<uint8_t> compress(const uint8_t* data, size_t size,
                                          int target_bits);

    /// 解压 KV Cache
    static std::vector<uint8_t> decompress(const uint8_t* data, size_t size,
                                            int original_bits);

    /// 估算压缩后的大小
    static size_t estimateCompressedSize(size_t original_size, int target_bits);

    /// 分页管理
    class PagedKVPool {
    public:
        struct Page {
            int page_id;
            int layer_id;
            int seq_start;
            int seq_end;
            void* device_ptr;
            bool is_pinned;       // 锁定在 GPU 内存
            bool is_dirty;        // 需要写回
        };

        PagedKVPool(size_t page_size, size_t max_pages);
        ~PagedKVPool();

        /// 分配一个页
        Page* allocate(int layer_id, int seq_start);

        /// 释放一个页
        void deallocate(int page_id);

        /// 换出到 CPU 内存
        bool evict(Page* page);

        /// 从 CPU 预取到 GPU
        bool prefetch(Page* page);

        /// 获取内存使用统计
        struct PoolStats {
            size_t total_pages;
            size_t used_pages;
            size_t gpu_pages;
            size_t cpu_pages;
            size_t total_memory;
            size_t gpu_memory;
            int eviction_count;
        };
        PoolStats getStats() const;

    private:
        struct PoolImpl;
        std::unique_ptr<PoolImpl> pool_impl_;
    };
};

} // namespace edgepilot
