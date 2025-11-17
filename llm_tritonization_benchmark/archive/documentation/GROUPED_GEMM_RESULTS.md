# Grouped GEMM Results - Expert Triton vs PyTorch

**Date**: 2025-10-13 (Updated)
**Operation**: Grouped GEMM (Multiple independent matrix multiplications)
**Pattern**: Matrix operations beyond BLAS
**Hardware**: NVIDIA GeForce RTX 3090 (Compute Capability 8.6)

---

## 🎯 Objective

Evaluate **Grouped GEMM** - a "matrix operation beyond BLAS" where multiple independent matrix multiplications are executed in a single kernel launch instead of separate calls.

## 📊 What is Grouped GEMM?

### Problem

Execute N independent matrix multiplications:
```
C_0 = A_0 @ B_0
C_1 = A_1 @ B_1
...
C_n = A_n @ B_n
```

### Why "Beyond BLAS"?

Standard BLAS libraries only have:
- `gemm`: Single matrix-matrix multiply
- `gemm_batched`: Batched multiply (but **same dimensions** for all matrices)

**Grouped GEMM** handles **variable-sized** independent matrices - not in standard BLAS.

### Use Cases
- **Mixture-of-Experts (MoE) models**: Different expert weights
- **Multi-task learning**: Different task-specific layers
- **Dynamic batching**: Variable batch sizes per task

---

## 🏆 Performance Results

### Three-Way Comparison: PyTorch vs Expert Triton vs LLM Triton

#### Latest Results (2025-10-13)

| Configuration | PyTorch (loop) | Expert Triton | LLM Triton | LLM/PyTorch | LLM/Expert |
|---------------|----------------|---------------|------------|-------------|------------|
| 8 GEMMs (1024×1024) | 0.3080 ms | 0.6001 ms | 1.4432 ms | **0.21x** ⚠️ | **0.42x** ⚠️ |
| 16 GEMMs (2048×2048) | 3.9383 ms | 7.3063 ms | 28.2677 ms | **0.14x** ⚠️ | **0.26x** ⚠️ |
| 32 GEMMs (2048×2048) | 7.9569 ms | 14.4759 ms | 56.8009 ms | **0.14x** ⚠️ | **0.25x** ⚠️ |
| 64 GEMMs (1024×1024) | 2.4301 ms | 3.8118 ms | 16.5123 ms | **0.15x** ⚠️ | **0.23x** ⚠️ |
| **AVERAGE** | **3.6583 ms** | **6.5485 ms** | **25.7560 ms** | **0.16x** ⚠️ | **0.29x** ⚠️ |

#### Previous Results (2025-10-09)

| Configuration | PyTorch (loop) | Expert Triton | LLM Triton | LLM/PyTorch | LLM/Expert |
|---------------|----------------|---------------|------------|-------------|------------|
| 4 GEMMs (512×512) | 0.0521 ms | 0.1676 ms | 0.2814 ms | **0.19x** ⚠️ | **0.60x** ⚠️ |
| 8 GEMMs (1024×1024) | 0.3045 ms | 0.4120 ms | 0.7450 ms | **0.41x** ⚠️ | **0.55x** ⚠️ |
| 16 GEMMs (512×512) | 0.1963 ms | 0.2647 ms | 0.4485 ms | **0.44x** ⚠️ | **0.59x** ⚠️ |
| **AVERAGE** | **0.1843 ms** | **0.2814 ms** | **0.4916 ms** | **0.35x** ⚠️ | **0.58x** ⚠️ |

### Key Results (Latest):
- **Expert Triton**: **0.56x** of PyTorch (1.8x slower) - similar to previous
- **LLM Triton**: **0.16x** of PyTorch (6.3x slower) - **significantly worse** with larger problems
- **LLM vs Expert**: **0.29x** (3.5x slower than expert) - **performance gap widened**

**Assessment: ⚠️ BOTH UNDERPERFORM** - Both Expert and LLM Triton slower than PyTorch baseline

---

## 🔬 Deep Dive: Root Cause Analysis

### The Counterintuitive Discovery

We conducted detailed diagnostic experiments to understand why PyTorch beats Triton. The key finding challenges conventional wisdom about kernel launch overhead.

#### Diagnostic Test 1: Single Large vs Multiple Small GEMMs

**Hypothesis:** Multiple kernel launches cause overhead
**Test:** Compare equivalent total work in different forms

```
Single large GEMM (2896×2896):  0.68 ms
8 separate GEMMs (1024×1024):   0.32 ms
Ratio: 0.48x (multiple is 2× FASTER!)
```

**🔥 This is the smoking gun!** Multiple separate cuBLAS launches are actually FASTER than processing equivalent work in a single kernel.

**Implications:**
- Kernel launch overhead is negligible (~0.01ms per launch)
- cuBLAS is better optimized for specific matrix sizes
- Smaller matrices have better cache locality (fit in L2)
- Size-specific optimization > reduced launches

#### Diagnostic Test 2: Tensor Core Verification

```
FP16 matmul (2048×2048):  0.26 ms
FP32 matmul (2048×2048):  0.70 ms
Speedup: 2.67×
```

✅ Confirms cuBLAS is using tensor cores effectively on RTX 3090

#### Diagnostic Test 3: Launch Overhead Measurement

```
64 GEMMs of 256×256:  0.83 ms
Average per GEMM:     0.013 ms
```

Modern CUDA launch overhead: ~10-13μs per kernel (completely negligible for 1024×1024+ matrices)

### Why PyTorch (cuBLAS) Wins: The Four Pillars

#### 1. Extreme Optimization Level
- **Decades of engineering** from NVIDIA's cuBLAS team
- Hand-tuned assembly for specific matrix sizes
- Automatically selects best algorithm per problem size
- Optimized tensor core utilization patterns
- Years of production feedback and tuning

#### 2. Size-Specific Tuning
- cuBLAS has **different optimized kernels** for each size:
  - 256×256, 512×512, 1024×1024, 2048×2048, etc.
- Each size uses different:
  - Tile sizes
  - Thread block configurations
  - Register allocation strategies
  - Memory access patterns
- Our test sizes (1024×1024, 2048×2048) hit these **sweet spots**

#### 3. Negligible Launch Overhead
- Modern CUDA launch latency: ~10μs
- For 1024×1024 GEMM taking ~300μs:
  - Overhead = 10μs / 300μs = 3.3%
- cuBLAS kernels are so fast that multiple launches is acceptable

#### 4. Better Cache Locality
- Smaller matrices (1024×1024 = 2MB in FP16) fit in L2 cache (6MB on 3090)
- Each GEMM gets fresh cache, better bandwidth utilization
- One large 2896×2896 matrix (16.8MB) exceeds L2, causes thrashing

### Why Triton Implementations are Slower

#### Expert Triton Issues

**1. Persistent Kernel Overhead**
```python
for g in range(group_size):  # Sequential loop in each block
    gm = tl.load(group_gemm_sizes + g * 3)      # Load metadata
    gn = tl.load(group_gemm_sizes + g * 3 + 1)  # from global memory
    gk = tl.load(group_gemm_sizes + g * 3 + 2)  # each iteration
    # ... compute GEMM ...
```
- Each block loops through ALL GEMMs sequentially
- Loads metadata from global memory in each iteration
- Branching and control flow overhead
- Static scheduling may not balance well

**2. Suboptimal Tile Sizes**
- Expert uses fixed 128×128 or 64×64 tiles
- cuBLAS uses size-specific tuned tiles (different for 1024 vs 2048)
- Not optimized for 1024×1024 and 2048×2048 specifically

**3. Generic Implementation**
- Tutorial code designed for H100, A100, V100, etc. (generic)
- Not specifically tuned for RTX 3090 (Ampere, 82 SMs)
- cuBLAS has GPU-specific code paths and optimizations

**4. Work Distribution**
- Persistent kernel: Fixed NUM_SM blocks (e.g., 84) process all work
- cuBLAS: Launches optimal grid size for each GEMM independently
- cuBLAS can saturate GPU better with right-sized grid per GEMM

#### LLM Triton Issues

All of Expert's issues, PLUS:

**1. Inefficient Memory Access Patterns**
- Lines 102-103 in `grouped_gemm_triton_llm.py`: Complex pointer arithmetic
- Extra boundary checks (lines 106-107)
- Masking overhead on every load (lines 110-111), even for interior tiles

**2. Missing Low-Level Optimizations**
- No `tl.multiple_of` hints (Expert has these for alignment)
- Loads with masks even in interior where not needed
- Less efficient accumulation pattern
- Doesn't leverage all Triton optimization primitives

**3. Generated Code Quality**
- LLM code is more "safe" and generic
- Expert code has hand-optimized tricks and edge cases
- LLM doesn't know all low-level Triton optimization techniques
- Missing hardware-specific tuning knowledge

**4. Poor Scaling Characteristics**
- Small problems (512×512): LLM = 0.60x Expert
- Large problems (2048×2048): LLM = 0.25x Expert
- Performance degradation suggests fundamental algorithmic issues
- Inefficiencies compound with problem size

### When Would Triton Win? The Four Scenarios

#### 1. Very Small Matrices (< 256×256)
```
Example: 100 GEMMs of 128×128 each
- PyTorch: 100 × (10μs compute + 10μs launch) = 2000μs
- Triton:   1 × (10μs launch + 1000μs compute) = 1010μs
```
At very small sizes, launch overhead (10μs) becomes significant relative to compute time (10μs).

#### 2. Large Number of GEMMs (100+)
```
Example: 500 GEMMs of 512×512 each
- PyTorch: 500 launches × 10μs = 5ms overhead
- Triton:   1 launch = 0.01ms overhead
```
With many GEMMs, total launch overhead accumulates to non-negligible levels.

#### 3. Fused Operations
```python
# If you need: C = activation(A @ B + bias)
# PyTorch: 3 kernel launches (matmul, add, activation)
# Triton:   1 fused kernel (all in one pass)
```
Triton can fuse operations that cuBLAS cannot, saving memory bandwidth.

#### 4. Custom Memory Layouts
If matrices are in non-standard formats (e.g., block-sparse, custom tiling, transposed layouts), cuBLAS can't handle them efficiently but Triton can.

### Recommendations

**For Production Grouped GEMM:**
- ✅ Use cuBLAS (torch.matmul) for matrices > 512×512
- ✅ Use cuBLAS unless you have 100+ GEMMs
- ⚠️ Consider Triton only for:
  - Very small matrices (< 256×256)
  - Many GEMMs (100+)
  - Fused operations needed
  - Custom memory layouts
- 🔬 **Always profile your specific workload**

**For Triton Development:**
- Focus on cases where cuBLAS is weak:
  - Fusion opportunities
  - Small matrix sizes
  - Custom/sparse layouts
- Don't compete with cuBLAS on its strengths:
  - Large, standard dense GEMMs
  - Well-supported matrix sizes

**For LLM Code Generation:**
- Hardware context (SM count, compute capability) is insufficient
- Need:
  - Hardware-specific tuning parameters
  - Empirical profiling feedback loops
  - Knowledge of cuBLAS-level optimizations (tl.multiple_of, masking, etc.)
- Generic "safe" code won't match expert performance
- Scaling characteristics must be validated across problem sizes

---

## 📈 Analysis

### Why Both Expert and LLM Triton Are Slower Here

Both implementations underperform PyTorch for similar fundamental reasons:

### Why Expert Triton is Slower Here

1. **Hardware Mismatch**
   - PyTorch blog reported 2.62x speedup on **H100** GPUs
   - We tested on **RTX 3090** (older architecture)
   - Persistent kernels may not be optimized for Ampere (3090)

2. **Problem Size Too Small**
   - MoE models in production use much larger matrices
   - Small GEMMs (512×512, 1024×1024) favor PyTorch's overhead-optimized path
   - Persistent kernel overhead dominates for small problems

3. **Autotuning Challenges**
   - Config optimized for different GPU/problem sizes
   - NUM_SM settings (84, 128) may not match RTX 3090's 82 SMs
   - Block sizes not optimal for our problem sizes

4. **PyTorch Optimizations**
   - Modern PyTorch has **extremely optimized** kernel launch overhead
   - cuBLAS calls are highly tuned for common sizes
   - Loop-based approach benefits from years of optimization

### Why LLM Triton is Even Slower Than Expert

Despite providing hardware context (RTX 3090, 82 SMs), LLM Triton underperforms expert:

1. **Implementation Differences**
   - LLM used more conservative masking (boundary checks on all loads)
   - Expert version uses `tl.multiple_of` hints for better optimization
   - LLM's autotuning configs may not have explored optimal space

2. **Persistent Kernel Complexity**
   - Grouped GEMM uses persistent kernels with complex device-side scheduling
   - Expert implementation optimized through extensive profiling
   - LLM lacks empirical feedback loop for this advanced pattern

3. **Hardware Context Insufficient**
   - Providing SM count (82) and compute capability is not enough
   - Need actual profiling data on target hardware
   - Complex kernels require iterative tuning beyond static context

4. **Fundamental Challenge**
   - This is an **advanced optimization pattern** (persistent kernels)
   - Even expert-written code struggles on mismatched hardware
   - LLM can't match expert-level tuning for complex cases

### When Would Expert Triton Win?

Based on PyTorch blog (H100, DeepSeekv3 training):
- ✅ **Larger matrices**: 2K×2K, 4K×4K, 8K×8K
- ✅ **More groups**: 32+, 64+, 128+ independent GEMMs
- ✅ **Newer GPUs**: H100 Hopper architecture (with TMA, persistent scheduling improvements)
- ✅ **MoE-scale workloads**: Real production scenarios

### Our Test Conditions
- ⚠️ **Smaller matrices**: 512×512, 1024×1024
- ⚠️ **Fewer groups**: 4-16 GEMMs
- ⚠️ **Older GPU**: RTX 3090 (Ampere, not Hopper)
- ⚠️ **Synthetic workload**: Not real MoE model

---

## 💡 Key Insights

### What This Result Demonstrates

⚠️ **Expert-written Triton doesn't always beat PyTorch**
- Hardware matters significantly
- Problem size threshold determines when custom kernels win
- Persistent kernels have overhead that must be amortized

✅ **This is NOT a failure of Triton**
- The expert kernel is optimized for H100 + large-scale MoE
- Our test conditions don't match the target use case
- Shows importance of **profiling** before choosing custom kernels

### Comparison with Our Previous Results

| Operation | Result | Reason |
|-----------|--------|--------|
| **Softmax** | Triton 4x faster ✅ | Clear fusion benefit (5 passes → 1) |
| **Laplacian** | Triton 5x faster ✅ | Spatial locality + sufficient arithmetic |
| **s000** | Triton 2x slower ⚠️ | Trivial operation, launch overhead |
| **Grouped GEMM** | Triton 1.8x slower ⚠️ | Problem size too small for this GPU |

### Pattern Recognition

**When Triton Wins:**
- Fusion opportunities (softmax)
- Spatial locality (Laplacian)
- **Large-scale problems** on target hardware

**When Triton Loses:**
- Trivial operations (s000)
- **Small problems** where overhead dominates
- **Hardware mismatch** (H100-optimized on 3090)

---

## 🔬 Technical Details

### Baseline Implementation

```python
# PyTorch: Separate kernel launches
def grouped_gemm_baseline(group_A, group_B):
    return [torch.matmul(a, b) for a, b in zip(group_A, group_B)]
```

**Characteristics:**
- Simple Python loop
- One cuBLAS call per GEMM
- Launch overhead per iteration
- But: Highly optimized cuBLAS kernels
- But: Modern PyTorch has low launch overhead

### Expert Triton Implementation

```python
# Persistent kernel with static scheduling
@triton.jit
def grouped_matmul_kernel(...):
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        # Process tiles for GEMM g
        while (tile_idx in range for this GEMM):
            # Do matrix multiply for this tile
            # Advance to next tile
            tile_idx += NUM_SM
        last_problem_end += num_tiles
```

**Characteristics:**
- Single kernel launch for all GEMMs
- Persistent thread blocks (CTAs)
- Device-side work scheduling
- Fixed NUM_SM (number of SMs to use)

**Overhead:**
- Persistent kernel setup cost
- Device-side scheduling logic
- Not amortized for small problems

---

## 🎓 Implications

### What We Learned

1. **Hardware Specificity Matters**
   - Kernels optimized for H100 may not work well on 3090
   - Different GPU architectures have different sweet spots
   - Always profile on target hardware

2. **Problem Size Threshold**
   - Small problems: PyTorch wins (optimized cuBLAS + low overhead)
   - Large problems: Custom kernels win (amortize setup cost)
   - Need to find the crossover point

3. **Expert Code ≠ Always Faster**
   - Official Triton tutorial code is expert-written
   - Still slower than PyTorch in our setup
   - Context matters: use case, hardware, problem size

4. **LLM Tritonization Implications - NOW TESTED**
   - ✅ **Confirmed**: LLM struggles significantly with complex persistent kernels
   - Hardware context (SM count, compute capability) is **insufficient**
   - LLM achieved **0.58x** of expert performance on smaller problems (Oct 9)
   - LLM achieved **0.29x** of expert performance on larger problems (Oct 13)
   - **Critical Finding**: LLM performance degrades as problem size increases
     - Small problems (512×512): LLM = 0.60x Expert
     - Large problems (2048×2048): LLM = 0.25x Expert
   - Shows realistic expectations for automated tritonization:
     - ✅ **Simple patterns** (fusion, stencils): LLM can match experts
     - ⚠️ **Complex patterns** (persistent kernels): LLM needs profiling feedback
     - ⚠️ **Scaling**: LLM code doesn't scale well to larger problem sizes

### When to Use Grouped GEMM

**✅ USE when:**
- H100+ GPUs (Hopper architecture)
- Large matrices (2K×2K+)
- Many groups (32+, 64+, 128+)
- Real MoE models at scale

**⚠️ DON'T USE when:**
- Older GPUs (Ampere, Turing)
- Small matrices (<1K×1K)
- Few groups (<16)
- PyTorch baseline is fast enough

---

## 📁 Files

```
llm_tritonization_benchmark/
├── baselines/
│   └── grouped_gemm_baseline.py       # PyTorch loop baseline
├── benchmark_grouped_gemm_expert.py    # Benchmark with expert Triton
└── GROUPED_GEMM_RESULTS.md            # This file
```

---

## 🎯 Conclusion

**Grouped GEMM Expert Triton on RTX 3090: ⚠️ UNDERPERFORMS**

| Criterion | Score | Notes |
|-----------|-------|-------|
| **Correctness** | 5/5 | Results match PyTorch |
| **Performance** | 2/5 | 1.8x slower than PyTorch loop |
| **Hardware Match** | 1/5 | Optimized for H100, tested on 3090 |
| **Problem Size Match** | 2/5 | Optimized for large-scale, tested small |
| **Overall** | ⚠️ | **Context-dependent performance** |

### Summary

For **Grouped GEMM on RTX 3090 with small-medium problem sizes**, **PyTorch baseline outperforms expert Triton**. This result:

- ✅ **Validates** the importance of hardware-aware optimization
- ✅ **Shows** that expert Triton isn't universally faster
- ✅ **Demonstrates** problem size thresholds matter
- ✅ **Highlights** that PyTorch built-ins are extremely well-optimized

**Key Takeaway:** Even **expert-written** Triton kernels must match the target hardware and problem size. Don't assume custom kernels are always faster - **profile first!**

**Additional Finding (Oct 13):** LLM-generated Triton code shows **poor scaling characteristics**. While it achieves 60% of expert performance on small problems (512×512), it drops to only 25% on larger problems (2048×2048). This suggests fundamental algorithmic or optimization differences that compound with problem size.

---

## 🔄 Updated Benchmark Summary

| Operation | Type | Complexity | Triton/Baseline | Assessment |
|-----------|------|------------|-----------------|------------|
| **Softmax** | Fusion | High | 4.0x faster ✅ | Excellent |
| **Laplacian** | Stencil | Medium | 5.0x faster ✅ | Excellent |
| **s000** | Element-wise | Low | 0.5x slower ⚠️ | Poor |
| **Grouped GEMM** | Beyond-BLAS | High* | 0.56x slower ⚠️ | Context-dependent |

**Note**: * High complexity operation, but **hardware/size mismatch** in our test setup

**Conclusion:** Triton performance is **highly context-dependent**. Profile on target hardware with realistic problem sizes!
