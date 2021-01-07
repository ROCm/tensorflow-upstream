#include "tensorflow/core/kernels/benchmark_helper.h"

#include <chrono>
#include <fstream>

namespace tensorflow {
void BenchmarkHelper::Start() {
  mutex_lock l(mu_);
  if (!is_running_) {
    reporter_thread_ = std::thread(ReportFunc, this);
    is_running_ = true;
  }
}

void BenchmarkHelper::Stop() {
  mutex_lock l(mu_);
  if (!stop_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    stop_ = true;
  }
}

void BenchmarkHelper::RecordTM(float ts) {
  int idx = ((int)(ts)) % 10;
  idx = idx >= 10 ? 9 : idx;
  ++time_recorder_[idx].second;
}

void BenchmarkHelper::ReportFunc(BenchmarkHelper* helper) {
  static std::ofstream stream("/tmp/blaze_report.log");
  stream.clear();
  while(!helper->stop_) {
    stream << "blaze kernel qps: " << helper->counter_ << "\n";
    stream << "time_value: ";
    for (auto& pair : helper->time_recorder_) {
      stream << "[" << pair.first << "," << pair.second << "]; ";
    }
    stream << "\n";
    helper->Clear();
    stream.flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }
}

}
