#ifndef TENSORFLOW_CORE_KERNELS_BENCHMARK_HELPER_H_
#define TENSORFLOW_CORE_KERNELS_BENCHMARK_HELPER_H_

#include <atomic>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
class BenchmarkHelper {
 public:
  static BenchmarkHelper& GetInstance() {
    static BenchmarkHelper instance;
    return instance;
  }

  void Add() { ++counter_; }

  void Start();

  void Stop();

  void RecordTM(float ts);
 private:
  BenchmarkHelper() {
    counter_ = 0;
    is_running_ = false;
    stop_ = false;
    std::vector<std::pair<std::string, std::atomic<int>>> recorder(kTimeSeg);
    time_recorder_ = std::move(recorder);
    for (int i = 0; i < kTimeSeg; ++i) {
      time_recorder_[i].first = std::to_string(i + 1);
      time_recorder_[i].second = 0;
    }
  }

  void Clear() {
    counter_ = 0;
    for (auto& pair : time_recorder_) {
      pair.second = 0;
    }
  }

  static void ReportFunc(BenchmarkHelper* helper);

 private:
  const int kTimeSeg = 10;
  std::atomic<uint64_t> counter_;
  std::vector<std::pair<std::string, std::atomic<int>>> time_recorder_;
  bool is_running_;
  mutex mu_;
  mutex stop_mu_;

  bool stop_;
  std::thread reporter_thread_;
};
}
#endif //end TENSORFLOW_CORE_KERNELS_BENCHMARK_HELPER_H_
