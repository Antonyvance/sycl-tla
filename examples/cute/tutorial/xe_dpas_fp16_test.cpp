/***************************************************************************************************
* Copyright (C) 2025 Intel Corporation, All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice, this
* list of conditions and the following disclaimer.
*
* 2. Redistributions in binary form must reproduce the above copyright notice,
* this list of conditions and the following disclaimer in the documentation
* and/or other materials provided with the distribution.
*
* 3. Neither the name of the copyright holder nor the names of its
* contributors may be used to endorse or promote products derived from
* this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
**************************************************************************************************/

#include <sycl/sycl.hpp>
#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <cute/tensor.hpp>

#include "cutlass/platform/platform.h"
#include "cutlass/util/sycl_event_manager.hpp"

#include <iostream>
#include <iomanip>

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

using namespace cute;

// Simple test kernel that uses XE_DPAS with FP16
void
test_dpas_kernel(half_t* d_A,         // 8x16 matrix A (FP16) - row major
                 half_t* d_B,         // 16x16 matrix B (FP16) - stored for DPAS (N x K)
                 float* d_C)          // 8x16 result C (FP32) - row major
{
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
  int tid = item.get_local_id(0);

  // Only use first 16 threads (one subgroup)
  if (tid >= 16) return;

  constexpr int M = 8;
  constexpr int N = 16;
  constexpr int K = 16;

  // DPAS atom type
  using MmaOp = XE_DPAS_TT<8, float, half_t, half_t>;

  // The DPAS requires data in Intel vector format
  using DVector = typename MmaOp::DVector;
  using AVector = typename MmaOp::AVector;
  using BVector = typename MmaOp::BVector;

  // Create register vectors
  AVector rA;
  BVector rB;
  DVector rC;

  // Load A from global memory (round-robin distribution across subgroup)
  constexpr int A_elem = sizeof(AVector) / sizeof(half_t);
  #pragma unroll
  for (int i = 0; i < A_elem; ++i) {
    int idx = tid + i * 16;
    rA[i] = (idx < M * K) ? d_A[idx] : half_t(0.0f);
  }

  // Load B from global memory (round-robin distribution)
  constexpr int B_elem = sizeof(BVector) / sizeof(half_t);
  #pragma unroll
  for (int i = 0; i < B_elem; ++i) {
    int idx = tid + i * 16;
    rB[i] = (idx < N * K) ? d_B[idx] : half_t(0.0f);
  }

  // Initialize C to zero
  #pragma unroll
  for (int i = 0; i < M; ++i) {
    rC[i] = 0.0f;
  }

  // Call DPAS: d = a*b + c
  MmaOp::fma(rC, rA, rB, rC);

  // Store results back to global memory
  #pragma unroll
  for (int i = 0; i < M; ++i) {
    int idx = tid + i * 16;
    if (idx < M * N) {
      d_C[idx] = rC[i];
    }
  }
}

class DpasTestKernel;

void test_dpas_fp16(sycl::queue& Q)
{
  std::cout << "Testing XE_DPAS with FP16 data type...\n";

  constexpr int M = 8;
  constexpr int N = 16;
  constexpr int K = 16;

  // Allocate host memory
  std::vector<half_t> h_A(M * K);
  std::vector<half_t> h_B(K * N);
  std::vector<float> h_C(M * N, 0.0f);
  std::vector<float> h_C_ref(M * N, 0.0f);

  // Initialize with simple test pattern
  // A matrix: row-major, each element = row_idx + col_idx * 0.1
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < K; j++) {
      h_A[i * K + j] = half_t(float(i) + float(j) * 0.1f);
    }
  }

  // B matrix: column-major for DPAS consumption (stored as N x K)
  // Each element = col_idx + row_idx * 0.1
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < K; i++) {
      h_B[j * K + i] = half_t(float(j) + float(i) * 0.1f);
    }
  }

  // Compute reference result on CPU
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < K; k++) {
        sum += float(h_A[i * K + k]) * float(h_B[j * K + k]);
      }
      h_C_ref[i * N + j] = sum;
    }
  }

  // Allocate device memory
  half_t* d_A = sycl::malloc_device<half_t>(M * K, Q);
  half_t* d_B = sycl::malloc_device<half_t>(N * K, Q);
  float* d_C = sycl::malloc_device<float>(M * N, Q);

  // Copy data to device
  Q.memcpy(d_A, h_A.data(), M * K * sizeof(half_t));
  Q.memcpy(d_B, h_B.data(), N * K * sizeof(half_t));
  Q.memcpy(d_C, h_C.data(), M * N * sizeof(float));
  Q.wait();

  // Launch kernel with single subgroup (16 threads)
  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;

  syclex::properties kernel_props {
    syclex::sub_group_size<16>,
    intelex::grf_size<256>
  };

  auto event = Q.parallel_for<DpasTestKernel>(
    sycl::nd_range<1>(sycl::range<1>(16), sycl::range<1>(16)), 
    kernel_props,
    [=](auto) {
      test_dpas_kernel(d_A, d_B, d_C);
    }
  );

  event.wait();

  // Copy results back
  Q.memcpy(h_C.data(), d_C, M * N * sizeof(float)).wait();

  // Verify results
  bool passed = true;
  float max_error = 0.0f;
  constexpr float tolerance = 1e-3f;  // FP16 has limited precision

  std::cout << "\nVerifying results...\n";
  std::cout << std::setprecision(4) << std::fixed;

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      int idx = i * N + j;
      float error = std::abs(h_C[idx] - h_C_ref[idx]);
      max_error = std::max(max_error, error);

      if (error > tolerance) {
        if (passed) {  // Print first few errors
          std::cout << "ERROR at (" << i << "," << j << "): "
                    << "got " << h_C[idx] 
                    << ", expected " << h_C_ref[idx]
                    << ", error=" << error << "\n";
        }
        passed = false;
      }
    }
  }

  std::cout << "\nMax error: " << max_error << "\n";
  std::cout << "Tolerance: " << tolerance << "\n";

  if (passed) {
    std::cout << "\n✓ TEST PASSED: DPAS FP16 results are correct!\n";
  } else {
    std::cout << "\n✗ TEST FAILED: DPAS FP16 results have errors!\n";
  }

  // Print sample of actual vs expected (first 4x4 block)
  std::cout << "\nSample results (first 4x4 block):\n";
  std::cout << "Actual (DPAS result):\n";
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      std::cout << std::setw(8) << h_C[i * N + j] << " ";
    }
    std::cout << "\n";
  }

  std::cout << "\nExpected (CPU reference):\n";
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      std::cout << std::setw(8) << h_C_ref[i * N + j] << " ";
    }
    std::cout << "\n";
  }

  // Print matrix shapes for clarity
  std::cout << "\nMatrix shapes:\n";
  std::cout << "A: " << M << "x" << K << " (FP16)\n";
  std::cout << "B: " << K << "x" << N << " (FP16, stored column-major)\n";
  std::cout << "C: " << M << "x" << N << " (FP32 accumulator)\n";

  // Cleanup
  sycl::free(d_A, Q);
  sycl::free(d_B, Q);
  sycl::free(d_C, Q);
}

int main(int argc, char** argv)
{
  std::cout << "===========================================\n";
  std::cout << "XE_DPAS FP16 Simple Test\n";
  std::cout << "===========================================\n\n";

  try {
    sycl::queue Q;
    
    std::cout << "Running on device: " 
              << Q.get_device().get_info<sycl::info::device::name>() 
              << "\n\n";

    test_dpas_fp16(Q);

  } catch (sycl::exception const& e) {
    std::cerr << "SYCL exception: " << e.what() << "\n";
    return 1;
  } catch (std::exception const& e) {
    std::cerr << "Exception: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
