cmake_minimum_required(VERSION 3.16)
project(EdgePilot VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Options
option(EDGEPILOT_BUILD_TESTS "Build tests" OFF)
option(EDGEPILOT_USE_EXECUTORCH "Enable ExecuTorch backend" OFF)
option(EDGEPILOT_USE_NCNN "Enable ncnn backend" OFF)

# Core library
add_subdirectory(core)
