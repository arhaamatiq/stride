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

if(STRIDE_SERVER)
  # PCRE2: 8-bit only, JIT on, no pcre2grep or tests.
  set(PCRE2_BUILD_PCRE2_8 ON CACHE BOOL "" FORCE)
  set(PCRE2_BUILD_PCRE2_16 OFF CACHE BOOL "" FORCE)
  set(PCRE2_BUILD_PCRE2_32 OFF CACHE BOOL "" FORCE)
  set(PCRE2_BUILD_PCRE2GREP OFF CACHE BOOL "" FORCE)
  set(PCRE2_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(PCRE2_SUPPORT_JIT ON CACHE BOOL "" FORCE)
  set(PCRE2_SHOW_REPORT OFF CACHE BOOL "" FORCE)
  set(PCRE2_STATIC_PIC ON CACHE BOOL "" FORCE)
  FetchContent_Declare(pcre2
    URL https://github.com/PCRE2Project/pcre2/releases/download/pcre2-10.47/pcre2-10.47.tar.bz2
    URL_HASH SHA256=47fe8c99461250d42f89e6e8fdaeba9da057855d06eb7fc08d9ca03fd08d7bc7)
  # cpp-httplib is one header; fetch it from the tag so the hash is stable.
  FetchContent_Declare(httplib
    URL https://raw.githubusercontent.com/yhirose/cpp-httplib/v0.59.0/httplib.h
    URL_HASH SHA256=dc1e4de3e0ef3a18f3a5818fab563eb3fa45d5d3bb1db527a8d389077ed5ee6c
    DOWNLOAD_NO_EXTRACT TRUE)
  FetchContent_MakeAvailable(pcre2 httplib)

  add_library(stride_tokenizer STATIC src/tokenizer.cpp)
  target_include_directories(stride_tokenizer PUBLIC src)
  target_link_libraries(stride_tokenizer PUBLIC pcre2-8 nlohmann_json::nlohmann_json)
  target_compile_options(stride_tokenizer PRIVATE -O2 -Wall -Wextra)

  add_executable(tokenize_cli tools/tokenize_cli.cpp)
  target_link_libraries(tokenize_cli PRIVATE stride_tokenizer)

  find_package(Threads REQUIRED)
  add_executable(stride_server src/server.cpp)
  target_include_directories(stride_server SYSTEM PRIVATE ${httplib_SOURCE_DIR})
  target_compile_definitions(stride_server PRIVATE CPPHTTPLIB_LISTEN_BACKLOG=1024)
  target_link_libraries(stride_server PRIVATE stride_core stride_tokenizer Threads::Threads)
  target_compile_options(stride_server PRIVATE -O2 -Wall -Wextra)
endif()
