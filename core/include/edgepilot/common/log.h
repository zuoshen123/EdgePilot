#pragma once

// EdgePilot 统一日志宏。
// D/I/W 级别仅在 -DEP_LOGGING=ON 时编译进二进制，E 级别恒保留。

#ifdef __ANDROID__
#include <android/log.h>

#define EP_LOG_TAG "EdgePilot"

#ifdef EP_LOGGING
#define EP_LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, EP_LOG_TAG, __VA_ARGS__)
#define EP_LOGI(...) __android_log_print(ANDROID_LOG_INFO,  EP_LOG_TAG, __VA_ARGS__)
#define EP_LOGW(...) __android_log_print(ANDROID_LOG_WARN,  EP_LOG_TAG, __VA_ARGS__)
#else
#define EP_LOGD(...) ((void)0)
#define EP_LOGI(...) ((void)0)
#define EP_LOGW(...) ((void)0)
#endif

#define EP_LOGE(...) __android_log_print(ANDROID_LOG_ERROR, EP_LOG_TAG, __VA_ARGS__)

#else  // 非 Android：退化为 stderr
#include <cstdio>
#ifdef EP_LOGGING
#define EP_LOGD(...) std::fprintf(stderr, "[EP][D] " __VA_ARGS__)
#define EP_LOGI(...) std::fprintf(stderr, "[EP][I] " __VA_ARGS__)
#define EP_LOGW(...) std::fprintf(stderr, "[EP][W] " __VA_ARGS__)
#else
#define EP_LOGD(...) ((void)0)
#define EP_LOGI(...) ((void)0)
#define EP_LOGW(...) ((void)0)
#endif
#define EP_LOGE(...) std::fprintf(stderr, "[EP][E] " __VA_ARGS__)
#endif
