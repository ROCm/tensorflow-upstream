#ifndef TENSORFLOW_COMPILER_JIT_CONTEXT_INDEX_POOL_
#define TENSORFLOW_COMPILER_JIT_CONTEXT_INDEX_POOL_
#include <unordered_map>
#include <unordered_set>
#include <deque>

#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace grappler {

class ContextIndexPool {
public:
  ContextIndexPool(int max_count) {
    CHECK(max_count > 0);
    VLOG(1) << "Init ContextIndexPool " << this << " with num=" << max_count;
    for (int i = 0; i < max_count; i++) {
      resources_.push_back(i);
    }
  }
  
  int acquire() {
    mutex_lock lock(resource_mu_);
    while(resources_.empty()) {
      // sleep 10 micro seconds to wait
      LOG(WARNING) << "Context pool is busy, waiting...";
      std::this_thread::sleep_for (std::chrono::microseconds(10));
    }

    int res = *resources_.begin();
    VLOG(1) << this << " acquire " << res;
    resources_.pop_front();
    return res;
  }

  void release(int val) {
    mutex_lock lock(resource_mu_);
    VLOG(1) << this << " release " << val;
    resources_.push_back(val);
  }

private:
  std::deque<int> resources_;
  mutex resource_mu_;
};


// User only need to acquire the next context index,
// and the wrapper will realease it automaticlly
class ContextIndexWrapper {
public:
  ContextIndexWrapper(std::shared_ptr<ContextIndexPool> pool) {
    CHECK(pool != nullptr);
    pool_ = pool;
  }

  int acquire() {
    context_index_ = pool_->acquire();
    return context_index_;
  }

  ~ContextIndexWrapper() {
    if (context_index_ < 0) return;

    pool_->release(context_index_);
  }
private:
  int context_index_ = -1;
  std::shared_ptr<ContextIndexPool> pool_;  
}; 

}  // end namespace grappler
}  // end namespace tensorflow
#endif // TENSORFLOW_COMPILER_JIT_CONTEXT_INDEX_POOL_
