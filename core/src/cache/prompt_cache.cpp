#include "edgepilot/cache/prompt_cache.h"
#include <algorithm>
#include <cstring>

namespace edgepilot {

// ============================================================
// PromptCacheManager 实现
// ============================================================

struct PromptCacheManager::Impl {
    Config config;
    std::list<CacheEntry> lru_list;  // 最近最少使用排序
    std::unordered_map<std::string, std::list<CacheEntry>::iterator> index;

    mutable std::mutex mutex;

    // 统计
    int total_lookups = 0;
    int cache_hits = 0;
    float total_match_ratio = 0.0f;
    int match_count = 0;

    size_t current_memory = 0;

    /// 计算 token 序列的 hash key
    static std::string hashTokens(const std::vector<int>& tokens, int max_len) {
        // 简单 hash: 取前 max_len 个 token 的 FNV-1a
        size_t hash = 14695981039346656037ULL;
        int len = std::min(static_cast<int>(tokens.size()), max_len);
        for (int i = 0; i < len; i++) {
            hash ^= static_cast<size_t>(tokens[i]);
            hash *= 1099511628211ULL;
        }
        return std::to_string(hash);
    }

    /// LRU 淘汰直到内存低于阈值
    void evictIfNeeded() {
        float usage = static_cast<float>(current_memory) / config.max_memory_bytes;
        while (usage > config.eviction_threshold && !lru_list.empty()) {
            auto& oldest = lru_list.back();
            current_memory -= oldest.memory_bytes;
            index.erase(oldest.entry_id);
            lru_list.pop_back();
            usage = static_cast<float>(current_memory) / config.max_memory_bytes;
        }
    }

    /// 查找最长前缀匹配
    CacheHit findBest(const std::vector<int>& tokens, int agent_id) {
        std::lock_guard<std::mutex> lock(mutex);
        total_lookups++;

        CacheHit best{};
        best.found = false;
        best.matched_length = 0;

        // 遍历所有缓存条目，找最长前缀匹配
        for (auto& entry : lru_list) {
            // 如果指定了 agent_id，只匹配同 Agent 的缓存
            if (agent_id >= 0 && entry.agent_id != agent_id) continue;

            int match_len = 0;
            int cmp_len = std::min(tokens.size(), entry.token_prefix.size());
            for (int i = 0; i < cmp_len; i++) {
                if (tokens[i] == entry.token_prefix[i]) {
                    match_len++;
                } else {
                    break;
                }
            }

            if (match_len > best.matched_length) {
                best.found = true;
                best.entry_id = entry.entry_id;
                best.matched_length = match_len;
                best.total_length = static_cast<int>(tokens.size());
                best.match_ratio = static_cast<float>(match_len) / tokens.size();
                best.entry = &entry;

                // 更新 LRU: 移到最前
                lru_list.splice(lru_list.begin(), lru_list, index[entry.entry_id]);
                entry.last_used = std::chrono::steady_clock::now();
                entry.use_count++;
            }
        }

        if (best.found) {
            cache_hits++;
            total_match_ratio += best.match_ratio;
            match_count++;
        }

        return best;
    }

    /// 存入缓存
    Status store(const std::vector<int>& tokens,
                 const std::vector<uint8_t>& kv_data, int agent_id) {
        std::lock_guard<std::mutex> lock(mutex);

        // 淘汰旧条目
        evictIfNeeded();

        // 创建新条目
        CacheEntry entry;
        entry.entry_id = hashTokens(tokens, static_cast<int>(tokens.size()));
        entry.token_prefix = tokens;
        entry.kv_data = kv_data;
        entry.prefix_length = static_cast<int>(tokens.size());
        entry.memory_bytes = kv_data.size();
        entry.created_at = std::chrono::steady_clock::now();
        entry.last_used = entry.created_at;
        entry.use_count = 0;
        entry.agent_id = agent_id;

        // 插入 LRU 头部
        lru_list.push_front(entry);
        index[entry.entry_id] = lru_list.begin();
        current_memory += entry.memory_bytes;

        return Status::OK;
    }
};

PromptCacheManager::PromptCacheManager()
    : impl_(std::make_unique<Impl>()) {}

PromptCacheManager::~PromptCacheManager() = default;

Status PromptCacheManager::initialize(const Config& config) {
    impl_->config = config;
    return Status::OK;
}

PromptCacheManager::CacheHit PromptCacheManager::findBestCache(
    const std::vector<int>& tokens, int agent_id) {
    return impl_->findBest(tokens, agent_id);
}

Status PromptCacheManager::store(const std::vector<int>& tokens,
                                  const std::vector<uint8_t>& kv_data,
                                  int agent_id) {
    return impl_->store(tokens, kv_data, agent_id);
}

void PromptCacheManager::invalidate(const std::string& entry_id) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto it = impl_->index.find(entry_id);
    if (it != impl_->index.end()) {
        impl_->current_memory -= it->second->memory_bytes;
        impl_->lru_list.erase(it->second);
        impl_->index.erase(it);
    }
}

void PromptCacheManager::invalidateForAgent(int agent_id) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto it = impl_->lru_list.begin();
    while (it != impl_->lru_list.end()) {
        if (it->agent_id == agent_id) {
            impl_->current_memory -= it->memory_bytes;
            impl_->index.erase(it->entry_id);
            it = impl_->lru_list.erase(it);
        } else {
            ++it;
        }
    }
}

void PromptCacheManager::clear() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->lru_list.clear();
    impl_->index.clear();
    impl_->current_memory = 0;
}

PromptCacheManager::CacheStats PromptCacheManager::getStats() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    CacheStats stats{};
    stats.total_entries = static_cast<int>(impl_->lru_list.size());
    stats.total_memory_bytes = impl_->current_memory;
    stats.total_lookups = impl_->total_lookups;
    stats.cache_hits = impl_->cache_hits;
    stats.hit_rate = (impl_->total_lookups > 0)
        ? static_cast<float>(impl_->cache_hits) / impl_->total_lookups
        : 0;
    stats.avg_match_ratio = (impl_->match_count > 0)
        ? impl_->total_match_ratio / impl_->match_count
        : 0;
    stats.memory_usage_ratio = (impl_->config.max_memory_bytes > 0)
        ? static_cast<float>(impl_->current_memory) / impl_->config.max_memory_bytes
        : 0;
    return stats;
}

// ============================================================
// KVCacheCompressor 实现 (简化版)
// ============================================================

std::vector<uint8_t> KVCacheCompressor::compress(const uint8_t* data,
                                                   size_t size,
                                                   int target_bits) {
    // TODO: 实际实现 INT4/INT8 量化压缩
    // 简化版: 返回原始数据的副本
    return std::vector<uint8_t>(data, data + size);
}

std::vector<uint8_t> KVCacheCompressor::decompress(const uint8_t* data,
                                                     size_t size,
                                                     int original_bits) {
    (void)original_bits;
    return std::vector<uint8_t>(data, data + size);
}

size_t KVCacheCompressor::estimateCompressedSize(size_t original_size,
                                                  int target_bits) {
    // FP16 → INT4: 压缩 4 倍
    // FP16 → INT8: 压缩 2 倍
    float ratio = 16.0f / target_bits;
    return static_cast<size_t>(original_size / ratio);
}

// ============================================================
// PagedKVPool 简化实现
// ============================================================

struct KVCacheCompressor::PagedKVPool::PoolImpl {
    size_t page_size;
    size_t max_pages;
    std::vector<Page> pages;
    int next_page_id = 0;
    int eviction_count = 0;
};

KVCacheCompressor::PagedKVPool::PagedKVPool(size_t page_size, size_t max_pages)
    : pool_impl_(std::make_unique<PoolImpl>()) {
    pool_impl_->page_size = page_size;
    pool_impl_->max_pages = max_pages;
}

KVCacheCompressor::PagedKVPool::~PagedKVPool() = default;

KVCacheCompressor::PagedKVPool::Page*
KVCacheCompressor::PagedKVPool::allocate(int layer_id, int seq_start) {
    if (pool_impl_->pages.size() >= pool_impl_->max_pages) {
        // 需要淘汰
        return nullptr;
    }

    Page page;
    page.page_id = pool_impl_->next_page_id++;
    page.layer_id = layer_id;
    page.seq_start = seq_start;
    page.seq_end = seq_start + static_cast<int>(pool_impl_->page_size);
    page.device_ptr = nullptr;  // TODO: 实际分配 GPU 内存
    page.is_pinned = false;
    page.is_dirty = false;

    pool_impl_->pages.push_back(page);
    return &pool_impl_->pages.back();
}

void KVCacheCompressor::PagedKVPool::deallocate(int page_id) {
    auto& pages = pool_impl_->pages;
    pages.erase(
        std::remove_if(pages.begin(), pages.end(),
                        [page_id](const Page& p) {
                            return p.page_id == page_id;
                        }),
        pages.end());
}

bool KVCacheCompressor::PagedKVPool::evict(Page* page) {
    if (!page || page->is_pinned) return false;
    // TODO: 实际将 GPU 内存换出到 CPU
    pool_impl_->eviction_count++;
    return true;
}

bool KVCacheCompressor::PagedKVPool::prefetch(Page* page) {
    if (!page) return false;
    // TODO: 实际从 CPU 预取到 GPU
    return true;
}

KVCacheCompressor::PagedKVPool::PoolStats
KVCacheCompressor::PagedKVPool::getStats() const {
    PoolStats stats{};
    stats.total_pages = pool_impl_->max_pages;
    stats.used_pages = pool_impl_->pages.size();
    stats.total_memory = pool_impl_->max_pages * pool_impl_->page_size;
    stats.eviction_count = pool_impl_->eviction_count;
    return stats;
}

} // namespace edgepilot
