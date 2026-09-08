cmake_minimum_required(VERSION 3.16)
project(edgepilot_core VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================
# llama.cpp (子模块)
# ============================================================
set(LLAMA_STANDALONE OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_UI OFF CACHE BOOL "" FORCE)

# 禁用 GPU 后端
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)

add_subdirectory(third_party/llama.cpp)

# ============================================================
# 源文件
# ============================================================
set(EDGEPILOT_SOURCES
    src/backend/inference_backend.cpp
    src/backend/backend_factory.cpp
    src/backend/ggml_backend.cpp
    src/decoder/speculative_decoder.cpp
    src/cache/prompt_cache.cpp
    src/scheduler/agent_scheduler.cpp
    src/metrics/metrics_collector.cpp
)

# ============================================================
# 核心库 (静态库)
# ============================================================
add_library(edgepilot_core STATIC ${EDGEPILOT_SOURCES})

target_include_directories(edgepilot_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/include
)

target_link_libraries(edgepilot_core PUBLIC llama)

# ============================================================
# 可选后端
# ============================================================
if(EDGEPILOT_USE_EXECUTORCH)
    find_package(executorch REQUIRED)
    target_link_libraries(edgepilot_core PRIVATE executorch)
    target_compile_definitions(edgepilot_core PRIVATE EDGEPILOT_HAS_EXECUTORCH=1)
endif()

if(EDGEPILOT_USE_NCNN)
    find_package(ncnn REQUIRED)
    target_link_libraries(edgepilot_core PRIVATE ncnn)
    target_compile_definitions(edgepilot_core PRIVATE EDGEPILOT_HAS_NCNN=1)
endif()

find_package(Threads REQUIRED)
target_link_libraries(edgepilot_core PUBLIC Threads::Threads)

if(ANDROID)
    target_link_libraries(edgepilot_core PUBLIC log)
endif()

# ============================================================
# JNI 绑定 (Android) + 16KB 页面对齐
# ============================================================
if(ANDROID)
    add_library(edgepilot_jni SHARED src/jni/edgepilot_jni.cpp)
    target_link_libraries(edgepilot_jni PRIVATE edgepilot_core)
    target_link_options(edgepilot_jni PRIVATE "LINKER:-z,max-page-size=16384")
endif()

# ============================================================
# 安装
# ============================================================
install(TARGETS edgepilot_core
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
install(DIRECTORY include/edgepilot DESTINATION include)
