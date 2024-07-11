// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>
#include <stdexcept>

#define HEAD_SZ_SWITCH(HEAD_SZ, CONST_NAME, ...)                 \
  [&] {                                                          \
    if (HEAD_SZ <= 32) {                                         \
      constexpr ck_tile::index_t CONST_NAME = 32;                \
      __VA_ARGS__();                                             \
    } else if (HEAD_SZ <= 64) {                                  \
      constexpr ck_tile::index_t CONST_NAME = 64;                \
      __VA_ARGS__();                                             \
    } else if (HEAD_SZ <= 128) {                                 \
      constexpr ck_tile::index_t CONST_NAME = 128;               \
      __VA_ARGS__();                                             \
    } else if (HEAD_SZ <= 256) {                                 \
      constexpr ck_tile::index_t CONST_NAME = 256;               \
      __VA_ARGS__();                                             \
    } else {                                                     \
      throw std::runtime_error("Head-dim sizes not supported!"); \
    }                                                            \
  }()
