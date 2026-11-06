cmake_minimum_required(VERSION 3.22)
project(stride LANGUAGES CXX)

option(STRIDE_CUDA "Build the CUDA backend (otherwise the CPU backend)" OFF)
option(STRIDE_MUTATION "Compile the bug-injection points used by the harness" OFF)
option(STRIDE_SERVER "Build the tokenizer and HTTP server" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)
FetchContent_Declare(nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d)
FetchContent_MakeAvailable(nlohmann_json)

# Reproducible floating point: no fast-math, no implicit FMA contraction. Dot
# products call fmaf explicitly, so the same operations run on every backend.
set(STRIDE_FP_FLAGS -fno-fast-math -ffp-contract=off -fno-strict-aliasing)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  list(APPEND STRIDE_FP_FLAGS -mfma -mavx2)
endif()

set(STRIDE_SOURCES
  src/safetensors.cpp
  src/model.cpp
  src/kv_cache.cpp
  src/engine.cpp
  src/scheduler.cpp)
set(STRIDE_KERNEL_SOURCES src/kernels.cu src/device.cu)

if(STRIDE_CUDA)
  # sm_89 is Ada (RTX 4090). -fmad=false: the compiler must not fuse a*b+c on its own.
  set(CMAKE_CUDA_ARCHITECTURES 89)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -fmad=false -Xcompiler=-fno-strict-aliasing")
  set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
  add_library(stride_core STATIC ${STRIDE_SOURCES} ${STRIDE_KERNEL_SOURCES})
  target_compile_definitions(stride_core PUBLIC STRIDE_BACKEND_CUDA)
else()
  set_source_files_properties(${STRIDE_KERNEL_SOURCES} PROPERTIES LANGUAGE CXX COMPILE_OPTIONS "-x;c++")
  add_library(stride_core STATIC ${STRIDE_SOURCES} ${STRIDE_KERNEL_SOURCES})
  find_package(Threads REQUIRED)
  target_link_libraries(stride_core PUBLIC Threads::Threads)
endif()

target_include_directories(stride_core PUBLIC src)
target_link_libraries(stride_core PUBLIC nlohmann_json::nlohmann_json)
target_compile_options(stride_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra ${STRIDE_FP_FLAGS}>)
if(STRIDE_MUTATION)
  target_compile_definitions(stride_core PUBLIC STRIDE_MUTATION)
endif()

add_executable(stride_cli tools/stride_cli.cpp)
target_link_libraries(stride_cli PRIVATE stride_core)
target_compile_options(stride_cli PRIVATE -O2 ${STRIDE_FP_FLAGS})
