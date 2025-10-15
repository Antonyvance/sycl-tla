/***************************************************************************************************
* Copyright (C) 2025 Intel Corporation, All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause
*
* Simple test demonstrating XE_DPAS with FP16 using CuTe abstractions
**************************************************************************************************/

#include <sycl/sycl.hpp>
#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <cute/tensor.hpp>

#include "cutlass/platform/platform.h"

#include <iostream>
#include <iomanip>

using namespace cute;

// Simple kernel using TiledMMA with DPAS
template <class TiledMMA>
void gemm_kernel(half_t const* A, half_t const* B, float* C, TiledMMA const& mma)
{
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
  int tid = item.get_local_id(0);

  // Constants for this simple test
  constexpr int M = 8;
  constexpr int N = 16;
  constexpr int K = 16;

  // Create global tensors (views)
  Tensor gA = make_tensor(make_gmem_ptr(A), make_layout(make_shape(Int<M>{}, Int<K>{})));  // 8x16
  Tensor gB = make_tensor(make_gmem_ptr(B), make_layout(make_shape(Int<N>{}, Int<K>{})));  // 16x16
  Tensor gC = make_tensor(make_gmem_ptr(C), make_layout(make_shape(Int<M>{}, Int<N>{})));  // 8x16

  // Get this thread's slice of the MMA
  auto thr_mma = mma.get_slice(tid);

  // Partition the MMA fragments
  auto tCrA = thr_mma.partition_fragment_A(gA);
  auto tCrB = thr_mma.partition_fragment_B(gB);
  auto tCrC = thr_mma.partition_fragment_C(gC);

  // Partition global memory for this thread
  auto tAgA = thr_mma.partition_A(gA);
  auto tBgB = thr_mma.partition_B(gB);
  auto tCgC = thr_mma.partition_C(gC);

  // Clear accumulator
  clear(tCrC);

  // Load A and B from global memory
  copy(tAgA, tCrA);
  copy(tBgB, tCrB);

  // Perform GEMM: C = A * B
  gemm(mma, tCrA, tCrB, tCrC);

  // Store C to global memory
  copy(tCrC, tCgC);
}

class DpasTestKernel;

void test_dpas_fp16(sycl::queue& Q)
{
  std::cout << "Testing XE_DPAS with FP16 using TiledMMA...\n";

  constexpr int M = 8;
  constexpr int N = 16;
  constexpr int K = 16;

  // Allocate host memory
  std::vector<half_t> h_A(M * K);
  std::vector<half_t> h_B(N * K);
  std::vector<float> h_C(M * N, 0.0f);
  std::vector<float> h_C_ref(M * N, 0.0f);

  // Initialize with simple test pattern
  // A matrix: row-major
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < K; j++) {
      h_A[i * K + j] = half_t(float(i) + float(j) * 0.1f);
    }
  }

  // B matrix: column-major (NxK layout)
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

  // Create TiledMMA with DPAS atom
  using MmaAtom = XE_DPAS_TT<8, float, half_t, half_t>;
  using AtomLayoutMNK = Layout<Shape<_1,_1,_1>>;
  using TiledMMA = TiledMMA<MMA_Atom<MmaAtom>, AtomLayoutMNK>;

  TiledMMA mma;

  // Launch kernel
  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;

  syclex::properties kernel_props {
    syclex::sub_group_size<16>,
    intelex::grf_size<256>
  };

/*
  8×16 Result Matrix C
┌────────────────┐
│  16 threads    │  ← All 16 threads cooperate
│  execute 1     │     to produce this result
│  DPAS together │
└────────────────┘

Thread ownership (simplified):
- Thread 0: owns C[0,0], C[0,1], ..., C[1,0], C[1,1], ... (round-robin)
- Thread 1: owns C[0,2], C[0,3], ..., C[1,2], C[1,3], ...
- ...
- Thread 15: owns specific elements based on layout

Subgroup Size = 16: Intel Xe GPUs use 16-lane SIMD subgroups. Each DPAS instruction is executed cooperatively by all 16 threads in a subgroup.

Single MMA Atom Per Subgroup: Each DPAS atom (like XE_DPAS_TT<8, float, half_t, half_t>) is executed by exactly 16 threads working together. The 8 in XE_DPAS_TT<8, ...> is the systolic depth, not the number of threads.

Data Distribution: The 16 threads in the subgroup collectively own the data for one DPAS operation:

Each thread holds a portion of the A, B, and C matrices in its registers
Data is distributed in a round-robin fashion: thread 0 owns elements 0, 16, 32, ...; thread 1 owns elements 1, 17, 33, ..., etc.
In Your Test:

*/

  // Single subgroup (16 threads)
  auto event = Q.parallel_for<DpasTestKernel>(
    sycl::nd_range<1>(sycl::range<1>(16), sycl::range<1>(16)), 
    kernel_props,
    [=](auto) {
      gemm_kernel(d_A, d_B, d_C, mma);
    }
  );

  event.wait();

  // Copy results back
  Q.memcpy(h_C.data(), d_C, M * N * sizeof(float)).wait();

  // Verify results
  bool passed = true;
  float max_error = 0.0f;
  constexpr float tolerance = 1e-2f;  // FP16 has limited precision

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

  // Cleanup
  sycl::free(d_A, Q);
  sycl::free(d_B, Q);
  sycl::free(d_C, Q);
}

int main(int argc, char** argv)
{
  std::cout << "===========================================\n";
  std::cout << "XE_DPAS FP16 Test with TiledMMA\n";
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
