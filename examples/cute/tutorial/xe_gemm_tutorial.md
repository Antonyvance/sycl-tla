# SYCL*TLA Xe GEMM Tutorial: Using New APIs with Xe2 Architecture

This tutorial explains how to implement high-performance GEMM kernels using SYCL*TLA's new CuTe APIs, specifically designed to expose and leverage Intel Xe GPU architecture features.

## Table of Contents

1. [Introduction](#introduction)
2. [Understanding Intel Xe Architecture](#understanding-intel-xe-architecture)
3. [New API Overview](#new-api-overview)
4. [Step-by-Step GEMM Implementation](#step-by-step-gemm-implementation)
5. [Xe2-Specific Optimizations](#xe2-specific-optimizations)
6. [Performance Considerations](#performance-considerations)
7. [Advanced Topics](#advanced-topics)

## Introduction

The `xe_gemm.cpp` example demonstrates a complete GEMM (General Matrix Multiply) kernel implementation using SYCL*TLA's redesigned CuTe library. This new architecture explicitly exposes Intel Xe GPU hardware features, enabling developers to write more efficient and predictable kernels.

### Key Improvements in New Architecture

The new design offers several advantages over the old hidden approach:

1. **Explicit SIMD Control**: Direct exposure of subgroup (16-lane SIMD) programming model
2. **Hardware-Mapped Operations**: Block 2D copy atoms that map directly to hardware instructions
3. **Accurate Thread Ownership**: `SubgroupTensor` class that clearly defines which thread owns which data
4. **Efficient Type Conversions**: `reorder()` operations that can fuse layout changes with type conversions
5. **Automatic Hardware Configuration**: Helper functions that select optimal copy operations

## Understanding Intel Xe Architecture

### SIMD vs SIMT Model

Unlike NVIDIA GPUs (SIMT - Single Instruction Multiple Thread), Intel Xe GPUs use a **SIMD** (Single Instruction Multiple Data) architecture:

- **Subgroup Size**: 16 threads (SIMD lanes) execute in lockstep
- **Thread Ownership**: Each thread owns multiple elements in a round-robin pattern
  - Thread 0 owns elements: 0, 16, 32, 48, ...
  - Thread 1 owns elements: 1, 17, 33, 49, ...
  - And so on...

### Key Hardware Features

#### 1. DPAS (Dot Product Accumulate Systolic)

Hardware matrix multiplication instructions optimized for various data types:

```cpp
// Example: 8x16x16 DPAS for half-precision
XE_DPAS_TT<8, float, half_t, half_t>
//         ^   ^      ^       ^
//         |   |      |       └─ B matrix element type
//         |   |      └───────── A matrix element type  
//         |   └──────────────── Accumulator type
//         └──────────────────── Systolic depth (8 or 16)
```

#### 2. Block 2D Memory Operations

Hardware-accelerated 2D block copy operations with automatic:
- Bounds checking
- Layout transformations (transpose, VNNI reordering)
- Prefetching capabilities

Five distinct block 2D copy atom types:
- `XE_LOAD_2D`: Standard 2D load
- `XE_STORE_2D`: Standard 2D store
- `XE_LOAD_VNNI_2D`: Load with VNNI layout transformation
- `XE_STORE_VNNI_2D`: Store with VNNI layout transformation
- `XE_PREFETCH_2D`: Prefetch to cache

#### 3. VNNI Layout (Vector Neural Network Instructions)

Special data layout that optimizes for DPAS operations:
- Reorders data to match DPAS consumption pattern
- Critical for achieving peak performance with narrow data types (int8, fp8, etc.)

## New API Overview

### Core Components

#### 1. TiledMMA - Matrix Multiply-Accumulate Orchestrator

```cpp
// Create TiledMMA using helper
auto mma = choose_tiled_mma(A, B, C);

// Components:
// - Workgroup tile size (e.g., 256x256xK)
// - Subgroup layout (e.g., 8x4 subgroups in n-major order)
// - DPAS atom configuration
```

#### 2. Block 2D Copy Helpers

High-level helpers that automatically select the right hardware operation:

```cpp
auto copy_a = make_block_2d_copy_A(mma, A);  // For A matrix
auto copy_b = make_block_2d_copy_B(mma, B);  // For B matrix  
auto copy_c = make_block_2d_copy_C(mma, C);  // For C matrix
```

These helpers:
- Analyze tensor layouts and data types
- Select appropriate block 2D copy atoms
- Configure VNNI transforms when beneficial
- Set up proper alignment and pitch

#### 3. SubgroupTensor

Accurately represents which SIMD lane owns which elements:

```cpp
auto tCrA = thr_mma.partition_sg_fragment_A(gA(_,_,0));
//   ^                                       ^
//   |                                       └─ Global tile shape
//   └───────────────────────────────────────── SubgroupTensor with accurate ownership
```

#### 4. Reorder Operations

Shuffle data between different layouts, with optional type conversion:

```cpp
reorder(tArA, tCrA);  // Copy layout → MMA layout
//      ^     ^
//      |     └─ Destination (MMA fragment layout)
//      └─────── Source (Copy fragment layout)
```

Can fuse type conversions (e.g., FP8 → FP16) with layout shuffles for efficiency.

## Step-by-Step GEMM Implementation

Let's walk through the `gemm_device()` function in `xe_gemm.cpp`:

### Step 1: Setup Workgroup Tiling

```cpp
// Get workgroup and thread IDs
auto item = sycl::ext::oneapi::this_work_item::get_nd_item<2>();
auto wg_m = int(item.get_group(1));
auto wg_n = int(item.get_group(0));
auto local_id = int(item.get_local_id(0));

// Create coordinate tensors (proxies for indexing)
Tensor cA = make_identity_tensor(A.shape());   // (M,K)
Tensor cB = make_identity_tensor(B.shape());   // (N,K)
Tensor cC = make_identity_tensor(C.shape());   // (M,N)

// Get workgroup tile size from MMA
auto wg_tile = mma.tile_mnk();  // e.g., (256, 256, 32)
auto wg_coord = make_coord(wg_m, wg_n, 0);

// Partition global tensors into workgroup tiles
Tensor gA = local_tile(cA, select<0,2>(wg_tile), make_coord(wg_m,_));  // (BLK_M,BLK_K,k)
Tensor gB = local_tile(cB, select<1,2>(wg_tile), make_coord(wg_n,_));  // (BLK_N,BLK_K,k)
Tensor gC = local_tile(cC, wg_tile, wg_coord, Step<_1,_1, X>{});       // (BLK_M,BLK_N)
```

**Key Points:**
- `local_tile()` creates a view of the global tensor for this workgroup
- `gA`, `gB`, `gC` are coordinate tensors (proxies), not actual data
- The last dimension of `gA` and `gB` represents k-tile iteration

### Step 2: Create Block 2D Copies

```cpp
// Create block 2D TiledCopies using helpers
auto copy_a = make_block_2d_copy_A(mma, A);
auto copy_b = make_block_2d_copy_B(mma, B);
auto copy_c = make_block_2d_copy_C(mma, C);
```

**What These Helpers Do:**
1. Analyze `mma` to determine optimal tile sizes
2. Inspect `A`, `B`, `C` layouts (row-major, column-major, VNNI, etc.)
3. Select appropriate hardware block 2D copy atoms
4. Configure transpose/VNNI transforms as needed
5. Return `TiledCopy` objects ready to use

**Xe2 Alignment:**
- Xe2's block 2D operations require specific alignment (typically 64-byte)
- Helpers automatically ensure compliance with hardware requirements
- VNNI transforms are applied when data types benefit (int8, fp8, etc.)

### Step 3: Partition to Thread Level

```cpp
// Slice TiledCopy/TiledMMA to this thread's work
auto thr_mma    =    mma.get_slice(local_id);
auto thr_copy_a = copy_a.get_slice(local_id);
auto thr_copy_b = copy_b.get_slice(local_id);
```

**Per-Thread Partitioning:**
- Each thread in the subgroup gets its slice of the operation
- Reflects the SIMD nature: 16 threads work cooperatively
- Thread ownership follows round-robin distribution

### Step 4: Create Register Fragments

```cpp
// Register fragments for MMA (what DPAS consumes)
auto tCrA = thr_mma.partition_sg_fragment_A(gA(_,_,0));
auto tCrB = thr_mma.partition_sg_fragment_B(gB(_,_,0));

// Register fragments for copies (what block 2D loads produce)
auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_,_,0));
auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_,_,0));

// Partition C for accumulation
Tensor tCrC = partition_fragment_C(mma, select<0,1>(wg_tile));
```

**Important Distinction:**
- `tArA`, `tBrB`: Copy destination layout (how block 2D loads organize data)
- `tCrA`, `tCrB`: MMA source layout (how DPAS expects data)
- These layouts differ! That's why we need `reorder()`

### Step 5: Partition Global Tensors for Copy

```cpp
// Partition global tensor proxies for copies
Tensor tAgA = thr_copy_a.partition_S(gA);
Tensor tBgB = thr_copy_b.partition_S(gB);

// Partition C for writing results
Tensor tCgC = thr_mma.partition_C(gC);  // Matches copy_c's source layout
```

### Step 6: Setup Prefetching

```cpp
// Create prefetch operations
auto prefetch_a = make_block_2d_prefetch(copy_a);
auto prefetch_b = make_block_2d_prefetch(copy_b);

auto thr_prefetch_A = prefetch_a.get_slice(local_id);
auto thr_prefetch_B = prefetch_b.get_slice(local_id);

// Partition for prefetch
auto pAgA = thr_prefetch_A.partition_S(gA);
auto pBgB = thr_prefetch_B.partition_S(gB);

const int prefetch_dist = 3;  // Prefetch 3 k-tiles ahead
```

**Xe2 Cache Optimization:**
- Prefetch brings data to L1 cache before use
- 3-tile distance balances latency hiding with cache pressure
- Uses dedicated `XE_PREFETCH_2D` hardware operation

### Step 7: Main Computation Loop

```cpp
int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));
int k_tile_prefetch = 0;

// Clear accumulators
clear(tCrC);

// Warm up prefetch pipeline
CUTE_UNROLL
for (; k_tile_prefetch < prefetch_dist; k_tile_prefetch++) {
  prefetch(prefetch_a, pAgA(_,_,_,k_tile_prefetch));
  prefetch(prefetch_b, pBgB(_,_,_,k_tile_prefetch));
}

// Main loop
for (int k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
  // Split barrier - first half
  barrier_arrive(barrier_scope);

  // Copy A/B from global (L1 cache) to registers
  copy(copy_a, tAgA(_,_,_,k_tile), tArA);
  copy(copy_b, tBgB(_,_,_,k_tile), tBrB);

  // Prefetch next tiles to L1
  prefetch(prefetch_a, pAgA(_,_,_,k_tile_prefetch));
  prefetch(prefetch_b, pBgB(_,_,_,k_tile_prefetch));

  // Shuffle from copy layout to MMA layout
  reorder(tArA, tCrA);
  reorder(tBrB, tCrB);

  // Accumulate C += A * B using DPAS
  gemm(mma, tCrA, tCrB, tCrC);

  // Split barrier - second half
  barrier_wait(barrier_scope);
}

// Write C to global memory
copy(copy_c, tCrC, tCgC);
```

**Key Operations Explained:**

#### `copy()` - Block 2D Load
```cpp
copy(copy_a, tAgA(_,_,_,k_tile), tArA);
```
- Executes block 2D load from global memory
- Hardware handles bounds checking, transpose if needed
- Loads into `tArA` with copy-optimized layout

#### `reorder()` - Layout Transformation
```cpp
reorder(tArA, tCrA);
```
- Shuffles data from copy layout to MMA layout
- Uses subgroup shuffle operations (hardware-accelerated)
- Can fuse type conversions (e.g., FP8→FP16) at no extra cost

#### `gemm()` - DPAS Execution
```cpp
gemm(mma, tCrA, tCrB, tCrC);
```
- Executes DPAS instructions
- Accumulates into `tCrC` (persistent across k-tiles)
- Single call may expand to multiple DPAS ops based on tile size

#### Split Barrier Pattern
```cpp
barrier_arrive(barrier_scope);
// ... compute work ...
barrier_wait(barrier_scope);
```
- Keeps threads loosely synchronized
- Allows overlap of independent operations
- `barrier_scope = 2` means workgroup-level synchronization

### Step 8: Write Results

```cpp
copy(copy_c, tCrC, tCgC);
```
- Uses block 2D store to write C to global memory
- Hardware handles bounds checking
- Optimal memory coalescing

## Xe2-Specific Optimizations

### 1. Choosing the Right DPAS Configuration

```cpp
template <typename TA, typename TB, typename TC>
auto choose_mma_op()
{
  if constexpr (is_complete_v<XE_DPAS_TT<8, TC, TA, TB>>)
    return XE_DPAS_TT<8, TC, TA, TB>{};
  else if constexpr (is_same_v<TA, cute::bfloat16_t>)
    return XE_DPAS_TT<8, float, cute::bfloat16_t>{};
  else
    return XE_DPAS_TT<8, float, cute::half_t>{};  // FP16 default
}
```

**Xe2 Considerations:**
- Systolic depth 8 preferred for most types on Xe2
- For narrow types (int8, fp8), systolic depth 16 may be better on Xe-HPC
- Type conversions: FP8/INT8 → FP16 often faster than → BF16

### 2. K-Dimension Tiling

```cpp
constexpr bool use_1x_dpas_per_k = 
  is_constant_v<1, decltype(stride<0>(A))> ||  // A^T case
  (byte && is_constant_v<1, decltype(stride<0>(B))>);  // Int8 B^N

using _K = conditional_t<use_1x_dpas_per_k,
                         C<op.K>,      // 1x DPAS in K
                         C<op.K*2>>;   // 2x DPAS in K
```

**Why This Matters:**
- Transposed A benefits from single DPAS in K dimension
- Byte types (int8, uint8, fp8) may need compiler improvements
- Balances register pressure with arithmetic intensity

### 3. Subgroup Layout

```cpp
using SGLayout = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;
//                             ^   ^   ^          ^   ^   ^
//                             M   N   K          N-major ordering
```

**Xe2 Optimization:**
- 8×4 subgroup tiling (32 subgroups total)
- N-major order improves memory coalescing
- Matches Xe2's execution unit layout

### 4. Large Workgroup Tiles

```cpp
using WGTile = Shape<_256, _256, _K>;  // 256x256 tiles
```

**Benefits on Xe2:**
- Amortizes workgroup launch overhead
- Better cache utilization
- Matches Xe2's large register file (256 GRF)

### 5. Type-Specific Optimizations

For FP8 types (float_e5m2_t, float_e4m3_t):

```cpp
// Reorder can fuse FP8→FP16 conversion with layout shuffle
reorder(tArA, tCrA);  // If tArA is FP8, automatically converts to FP16
```

**Xe2 Advantage:**
- Hardware accelerated FP8→FP16 conversion via DPAS
- Zero extra cost when fused with reorder
- Optimized VNNI reordering for narrow types

## Performance Considerations

### 1. Memory Alignment

- Block 2D operations require 64-byte alignment
- Helpers automatically pad/adjust for compliance
- Misalignment triggers slower scalar loads

### 2. Cache Hierarchy

```
L1 Cache (per XVE) → L2 Cache (shared) → HBM
     ^                    ^                 ^
     |                    |                 |
  Prefetch            Block 2D          Initial
  target              loads             data
```

- Prefetch distance = 3 tiles (tunable)
- Balance: too small = cache misses, too large = cache thrashing

### 3. Register Pressure

Xe2 GPU has 256 GRF (General Register File) per thread:
- Large tiles need more registers
- Monitor occupancy with profiling tools
- Reduce tile size if occupancy drops

### 4. Arithmetic Intensity

```
Arithmetic Intensity = FLOPs / Bytes Transferred
```

For 256×256 workgroup tiles with K=32:
- FLOPs: 2 × 256 × 256 × 32 = 4,194,304
- Bytes (FP16): (256×32 + 256×32) × 2 = 32,768
- AI = 128 FLOPs/Byte (excellent!)

## Advanced Topics

### Mixed Precision GEMM

```cpp
// Example: FP16 × FP8 → FP32
test_case<half_t, float_e4m3_t, float, 'R', 'C'>(Q, m, n, k);
```

How it works:
1. Load FP16 A matrix (no conversion needed)
2. Load FP8 B matrix with `XE_LOAD_VNNI_2D`
3. Reorder FP8→FP16 during shuffle (fused)
4. DPAS accumulates in FP32

### Custom Layouts

For unusual layouts, bypass helpers and create copies manually:

```cpp
auto copy_atom = XE_LOAD_2D<TA, decltype(shape<0>(wg_tile_a)), 
                                decltype(shape<1>(wg_tile_a))>{};
auto copy_a = make_tiled_copy(Copy_Atom<decltype(copy_atom)>{},
                              sg_layout, tile_layout);
```

### Epilogue Fusion

Extend the kernel to fuse operations after GEMM:

```cpp
// After gemm()
CUTE_UNROLL
for (int i = 0; i < size(tCrC); ++i) {
  tCrC(i) = relu(tCrC(i));  // Fused ReLU
}
copy(copy_c, tCrC, tCgC);
```

### Group-wise Quantization

For INT4/INT8 with scale factors:

```cpp
// Load quantized data
copy(copy_b, tBgB(_,_,_,k_tile), tBrB_int8);

// Reorder + dequantize
reorder_dequant(tBrB_int8, scales, tCrB_fp16);

// DPAS with dequantized data
gemm(mma, tCrA, tCrB_fp16, tCrC);
```

## Summary

The new SYCL*TLA CuTe APIs provide:

1. **Explicit Hardware Exposure**: Direct access to DPAS, block 2D ops, VNNI
2. **Accurate Ownership Model**: SubgroupTensor reflects SIMD lane ownership
3. **Efficient Abstractions**: Helpers automate hardware parameter selection
4. **Xe2 Optimization**: Designed specifically for Intel Xe architecture features
5. **Composability**: Mix and match atoms for custom kernels

### Key Takeaways

- Use `make_block_2d_copy_{A,B,C}()` helpers for automatic optimization
- Understand copy layout vs MMA layout - they differ!
- `reorder()` is critical for shuffling between layouts
- Prefetching hides memory latency on Xe2
- Type conversions fuse with reorders for free
- 256×256 tiles work well on Xe2 GPUs

### Next Steps

1. Experiment with different tile sizes for your problem
2. Profile using VTune or other tools
3. Explore mixed-precision configurations
4. Implement custom epilogues for your workload
5. Read the [Xe Rearchitecture Documentation](../../../media/docs/cpp/xe_rearchitecture.md)

## References

- [Xe Rearchitecture Design Doc](../../../media/docs/cpp/xe_rearchitecture.md)
- [CuTe Quick Start Guide](../../../media/docs/cpp/cute/00_quickstart.md)
- [Intel Xe-HPG/Xe2 Architecture](https://www.intel.com/content/www/us/en/architecture-and-technology/visual-technology/arc-graphics.html)
- SYCL*TLA Examples: `/examples/cute/tutorial/`

---

**Copyright (c) 2025 Intel Corporation. All rights reserved.**
**SPDX-License-Identifier: BSD-3-Clause**
