#pragma once
// FP8 header version 0.3, 2021/05/11

#define HIP_HOST_DEVICE __host__ __device__

namespace hip_f8_impl {

template <int wm, int we, typename T, bool negative_zero_nan, bool clip>
HIP_HOST_DEVICE
uint8_t cast_to_f8(T _x, bool stoch = false, uint32_t rng = 0);

template <int wm, int we, typename T, bool negative_zero_nan>
HIP_HOST_DEVICE
T cast_from_f8(uint8_t x);

} // namespace hip_f8_impl

#include "hip_f8_impl.h"

enum class hip_f8_type {
  bf8 = 0, // 1:5:2
  fp8 = 1  // 1:4:3
};


enum class hip_f8_rounding_mode {
  standard,
  stochastic
};


// bias mode bit implementation
//
// For MI100 simulation purpose, we keep a copy of it on the host and device
// (MI300 HW implementation will be different)
//
// The bias mode should only be accessed via its get/set routines.
// The set routine sets both copies to the same value, keeping them in sync
// The get routine will return the device copy for device functions and
// the host copy for host functions
//
// "bias mode optimial"
//    => "bias mode bit" = 1
//    => bias = 16 for 152, 8 for 143
//    => NAN/INF are represented as negative_zero
//
// "bias mode ieee"
//    => "bias mode bit" = 0
//    => bias = 15 for 152, 7 for 143
//    => NAN/INF are represented as per IEEE conventions

__device__ bool hip_f8_bias_mode_bit_device;
bool hip_f8_bias_mode_bit_host;

__global__ void set_hip_f8_bias_mode_bit(bool v) {
  hip_f8_bias_mode_bit_device = v;
}

void set_hip_f8_bias_mode_ieee() {
  hipLaunchKernelGGL(set_hip_f8_bias_mode_bit, dim3(1), dim3(1), 0, 0, false);
  hip_f8_bias_mode_bit_host = false;
}

void set_hip_f8_bias_mode_optimal() {
  hipLaunchKernelGGL(set_hip_f8_bias_mode_bit, dim3(1), dim3(1), 0, 0, true);
  hip_f8_bias_mode_bit_host = true;
}

inline HIP_HOST_DEVICE bool get_hip_f8_bias_mode() {
#if defined(__HIP_DEVICE_COMPILE__)
  return hip_f8_bias_mode_bit_device;

#else
  return hip_f8_bias_mode_bit_host;
#endif
}


template<hip_f8_type T>
struct hip_f8 {
  uint8_t data;

  // default constructor
  HIP_HOST_DEVICE hip_f8() = default;

  // constructor from bits
  explicit HIP_HOST_DEVICE hip_f8(uint8_t v) {
    data = v;
  }

  // constructor from float
  explicit HIP_HOST_DEVICE hip_f8(float v, hip_f8_rounding_mode rm=hip_f8_rounding_mode::standard, uint32_t rng=0) {
    if (T == hip_f8_type::bf8) {
      if (get_hip_f8_bias_mode()) {
	data = hip_f8_impl::cast_to_f8<2, 5, float, true/*negative_zero_nan*/, true/*clip*/>(v, (rm == hip_f8_rounding_mode::stochastic), rng);
      } else {
	data = hip_f8_impl::cast_to_f8<2, 5, float, false/*negative_zero_nan*/, true/*clip*/>(v, (rm == hip_f8_rounding_mode::stochastic), rng);
      }
    } else /* fp8*/ {
      if (get_hip_f8_bias_mode()) {
	data = hip_f8_impl::cast_to_f8<3, 4, float, true/*negative_zero_nan*/, true/*clip*/>(v, (rm == hip_f8_rounding_mode::stochastic), rng);
      } else {
	data = hip_f8_impl::cast_to_f8<3, 4, float, false/*negative_zero_nan*/, true/*clip*/>(v, (rm == hip_f8_rounding_mode::stochastic), rng);
      }
    }
  }

  // constructor from half
  explicit HIP_HOST_DEVICE hip_f8(half v, hip_f8_rounding_mode rm=hip_f8_rounding_mode::standard, uint32_t rng=0) {
    if (T == hip_f8_type::bf8) {
      if (get_hip_f8_bias_mode()) {
	data = hip_f8_impl::cast_to_f8<2, 5, half, true/*negative_zero_nan*/, true/*clip*/>(v, (rm == hip_f8_rounding_mode::stochastic), rng);
      } else {
	data = hip_f8_impl::cast_to_f8<2, 5, half, false/*negative_zero_nan*/, true/*clip*/>(v, (rm == hip_f8_rounding_mode::stochastic), rng);
      }
    } else /* fp8*/ {
      if (get_hip_f8_bias_mode()) {
	data = hip_f8_impl::cast_to_f8<3, 4, half, true/*negative_zero_nan*/, true/*clip*/>(v, (rm == hip_f8_rounding_mode::stochastic), rng);
      } else {
	data = hip_f8_impl::cast_to_f8<3, 4, half, false/*negative_zero_nan*/, true/*clip*/>(v, (rm == hip_f8_rounding_mode::stochastic), rng);
      }
    }
  }

  // constructor from hip_bfloat16
  explicit HIP_HOST_DEVICE hip_f8(hip_bfloat16 v, hip_f8_rounding_mode r=hip_f8_rounding_mode::standard, uint32_t rng=0);

  // convert to float
  explicit inline HIP_HOST_DEVICE operator float() const {
    if (T == hip_f8_type::bf8) {
      if (get_hip_f8_bias_mode()) {
	return hip_f8_impl::cast_from_f8<2, 5, float, true/*negative_zero_nan*/>(data);
      } else {
	return hip_f8_impl::cast_from_f8<2, 5, float, false/*negative_zero_nan*/>(data);
      }
    } else /* fp8*/ {
      if (get_hip_f8_bias_mode()) {
	return hip_f8_impl::cast_from_f8<3, 4, float, true/*negative_zero_nan*/>(data);
      } else {
	return hip_f8_impl::cast_from_f8<3, 4, float, false/*negative_zero_nan*/>(data);
      }
    }
  }

  // convert to half
  explicit inline HIP_HOST_DEVICE operator half() const {
    if (T == hip_f8_type::bf8) {
      if (get_hip_f8_bias_mode()) {
	return hip_f8_impl::cast_from_f8<2, 5, half, true/*negative_zero_nan*/>(data);
      } else {
	return hip_f8_impl::cast_from_f8<2, 5, half, false/*negative_zero_nan*/>(data);
      }
    } else /* fp8*/ {
      if (get_hip_f8_bias_mode()) {
	return hip_f8_impl::cast_from_f8<3, 4, half, true/*negative_zero_nan*/>(data);
      } else {
	return hip_f8_impl::cast_from_f8<3, 4, half, false/*negative_zero_nan*/>(data);
      }
    }
  }

  // convert to hip_bfloat16
  explicit inline HIP_HOST_DEVICE operator hip_bfloat16() const;

  // check for zero
  inline HIP_HOST_DEVICE bool is_zero() const {
    if (get_hip_f8_bias_mode()) {
      return data == 0x00;
    } else {
      return (data == 0x00) || (data == 0x80);
    }
  }
  
  // check for nan
  inline HIP_HOST_DEVICE bool is_nan() const {
    if (get_hip_f8_bias_mode()) {
      return data == 0x80;
    } else {
      if (T == hip_f8_type::bf8) {
	return
	  (data == 0x7d) || (data == 0x7e) || (data == 0x7f) ||
	  (data == 0xfd) || (data == 0xfe) || (data == 0xff);
      } else {
	return
	  (data == 0x79) || (data == 0x7a) || (data == 0x7b) || (data == 0x7c) || (data == 0x7d) || (data == 0x7e) || (data == 0x7f) ||
	  (data == 0xf9) || (data == 0xfa) || (data == 0xfb) || (data == 0xfc) || (data == 0xfd) || (data == 0xfe) || (data == 0xff);
      }
    }
  }
  
  // check for inf
  inline HIP_HOST_DEVICE bool is_inf() const {
    if (get_hip_f8_bias_mode()) {
      return data == 0x80;
    } else {
      if (T == hip_f8_type::bf8) {
	return (data == 0x7c) || (data == 0xfc);
      } else {
	return (data == 0x78) || (data == 0xf8);
      }
    }
  }
};

template<hip_f8_type T>
struct hip_f8x4 {
  // define some convenience types
  typedef float float32x2 __attribute__((ext_vector_type(2)));
  typedef float float32x4 __attribute__((ext_vector_type(4)));

  typedef _Float16 halfx2 __attribute__((ext_vector_type(2)));
  typedef _Float16 halfx4 __attribute__((ext_vector_type(4)));

  typedef uint16_t hip_bfloat16x2 __attribute__((ext_vector_type(2)));
  typedef uint16_t hip_bfloat16x4 __attribute__((ext_vector_type(4)));

  uint32_t data;

  // default constructor
  HIP_HOST_DEVICE hip_f8x4() = default;

  // constructor from bits
  HIP_HOST_DEVICE hip_f8x4(uint32_t v);

  // constructor from float
  HIP_HOST_DEVICE hip_f8x4(float v0, float v1=0, float v2=0, float v3=0, hip_f8_rounding_mode rm=hip_f8_rounding_mode::standard, uint32_t rng=0);
  HIP_HOST_DEVICE hip_f8x4(float32x2 v, hip_f8_rounding_mode rm=hip_f8_rounding_mode::standard, uint32_t rng=0);
  HIP_HOST_DEVICE hip_f8x4(float32x4 v, hip_f8_rounding_mode rm=hip_f8_rounding_mode::standard, uint32_t rng=0);

  // constructor from half
  HIP_HOST_DEVICE hip_f8x4(half v0, half v1=0, half v2=0, half v3=0, hip_f8_rounding_mode rm=hip_f8_rounding_mode::standard, uint32_t rng=0);
  HIP_HOST_DEVICE hip_f8x4(halfx2 v, hip_f8_rounding_mode rm=hip_f8_rounding_mode::standard, uint32_t rng=0);
  HIP_HOST_DEVICE hip_f8x4(halfx4 v, hip_f8_rounding_mode rm=hip_f8_rounding_mode::standard, uint32_t rng=0);

  // constructor from hip_bfloat16
  HIP_HOST_DEVICE hip_f8x4(hip_bfloat16 v0, hip_bfloat16 v1=hip_bfloat16(0.0f), hip_bfloat16 v2=hip_bfloat16(0.0f), hip_bfloat16 v3=hip_bfloat16(0.0f), hip_f8_rounding_mode rm=hip_f8_rounding_mode::standard, uint32_t rng=0);
  HIP_HOST_DEVICE hip_f8x4(hip_bfloat16x2 v, hip_f8_rounding_mode rm=hip_f8_rounding_mode::standard, uint32_t rng=0);
  HIP_HOST_DEVICE hip_f8x4(hip_bfloat16x4 v, hip_f8_rounding_mode rm=hip_f8_rounding_mode::standard, uint32_t rng=0);

  // convert to float32x4
  inline HIP_HOST_DEVICE operator float32x4() const;

  // convert to halfx4
  inline HIP_HOST_DEVICE operator halfx4() const;

  // convert to hip_bfloat16x4
  inline HIP_HOST_DEVICE operator hip_bfloat16x4() const;
};



template<hip_f8_type T>
struct hip_f8x8 {
  // define some convenience types
  typedef hip_f8x4<T>  f8x8 __attribute__((ext_vector_type(2)));

  f8x8 data;

  // default constructor
  HIP_HOST_DEVICE hip_f8x8() = default;

  // do we need to define other constructors or any conversion routines here?
};

// If we do not end up needing either any constructors or conversion routines for the above type, then
// we can simplify the above type to the following
#if USE_SIMPLER_HIP_F8x8
template <hip_f8_type T>
using hip_f8x8 = hip_f8x4<T> __attribute__((ext_vector_type(2)));
#endif

typedef float hip_float32x4  __attribute__((ext_vector_type(4)));
typedef float hip_float32x16 __attribute__((ext_vector_type(16)));

// these are device-specific and we don't expect them to exist unless we're compiling with hip-clang for MI300.
template<hip_f8_type T_A, hip_f8_type T_B>
__device__ hip_float32x4 mfma_f32_16x16x32(hip_f8x8<T_A> a, hip_f8x8<T_B> b, hip_float32x4 c);

template<hip_f8_type T_A, hip_f8_type T_B>
__device__ hip_float32x16 mfma_f32_32x32x16(hip_f8x8<T_A> a, hip_f8x8<T_B> b, hip_float32x16 c);

