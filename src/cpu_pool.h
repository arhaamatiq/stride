// Minimal fork-join pool for the CPU backend.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace stride {

class CpuPool {
 public:
  static CpuPool& instance() {
    static CpuPool pool;
    return pool;
  }

  void run(int n, const std::function<void(int)>& fn) {
    if (workers_.empty() || n == 1) {
      for (int i = 0; i < n; ++i) fn(i);
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mu_);
      fn_ = &fn;
      next_.store(0);
      total_ = n;
      pending_ = static_cast<int>(workers_.size());
      ++generation_;
    }
    cv_.notify_all();
    drain();
    std::unique_lock<std::mutex> lock(mu_);
    done_.wait(lock, [this] { return pending_ == 0; });
  }

 private:
  CpuPool() {
    int count = static_cast<int>(std::thread::hardware_concurrency());
    if (const char* env = std::getenv("STRIDE_CPU_THREADS")) count = std::atoi(env);
    for (int i = 0; i + 1 < count; ++i) workers_.emplace_back([this] { loop(); });
    for (auto& w : workers_) w.detach();
  }

  void drain() {
    for (;;) {
      const int i = next_.fetch_add(1);
      if (i >= total_) return;
      (*fn_)(i);
    }
  }

  void loop() {
    uint64_t seen = 0;
    for (;;) {
      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&] { return generation_ != seen; });
        seen = generation_;
      }
      drain();
      {
        std::lock_guard<std::mutex> lock(mu_);
        if (--pending_ == 0) done_.notify_all();
      }
    }
  }

  std::vector<std::thread> workers_;
  std::mutex mu_;
  std::condition_variable cv_, done_;
  const std::function<void(int)>* fn_ = nullptr;
  std::atomic<int> next_{0};
  int total_ = 0;
  int pending_ = 0;
  uint64_t generation_ = 0;
};

template <class F>
void cpu_parallel_for(int n, F&& f) {
  const std::function<void(int)> fn = std::forward<F>(f);
  CpuPool::instance().run(n, fn);
}

}  // namespace stride
