// Device memory and synchronization. One implementation per backend.
#pragma once

#include <cstddef>
#include <string>

namespace stride {

void device_init();
void* device_alloc(size_t bytes);
void device_free(void* ptr);
void device_upload(void* dst, const void* src, size_t bytes);
void device_download(void* dst, const void* src, size_t bytes);
void device_wait();
std::string device_name();

}  // namespace stride
