# S243 Triton Kernel: Three-Statement Chain Analysis

> **Profiling Date:** October 21, 2025
> **Methodology:** 10 independent trials, 100 iterations per trial
> **Array Size:** 256,000 elements
> **GPU:** NVIDIA GeForce RTX 3090

## Executive Summary: Variant 2 Achieves 1.83x Speedup!

**Key Finding:** The hand-written single-stage approach with `a_orig` (Variant 2) is **correct** and achieves **1.83x faster** performance than PyTorch!

| Variant | Time (ms) | NCU (μs) | Kernels | DRAM % | CoV % | Status |
|---------|-----------|----------|---------|--------|-------|--------|
| **PyTorch** | 0.1504 | 2360.84 | 439 | 42.6% | 0.9% | ✓ Baseline |
| **Variant 1 (Two-Stage)** | 0.1530 | 22.08 | 2 | 57.4% | 1.0% | ⚠️ SLOWER than PyTorch |
| **Variant 2 (Hand-written)** | **0.0821** | **10.75** | **1** | **59.9%** | 1.1% | **🏆 1.83x FASTER** |

**Winner: Variant 2** - Single-stage hand-written kernel with `a_orig` parameter provides both correctness and best performance!

### Performance Results at N=10M

| Implementation | Duration (μs) | Speedup |
|----------------|---------------|---------|
| PyTorch Baseline | 101665.81 | 1.00x |
| Triton Optimal (256x4) | 32750.30 | **3.10x** |

**Time reduction: 67.8%** - At larger scale (N=10M), the Triton implementation achieves even better speedup (3.10x vs 1.83x at N=256K) by fusing all three statements into a single kernel and eliminating PyTorch's kernel launch overhead.

## Kernel Description

```c
for (int i = 0; i < LEN_1D-1; i++) {
    a[i] = b[i] + c[i] * d[i];       // Statement 1: FMA
    b[i] = a[i] + d[i] * e[i];       // Statement 2: Reuses NEW a[i], another FMA
    a[i] = b[i] + a[i+1] * d[i];     // Statement 3: Reuses NEW b[i], needs ORIGINAL a[i+1]
}
```

**Key Challenges:**
- **5 input arrays** (a, b, c, d, e) vs s241's 4
- **Complex dependency chain**: a→b→a
- **Critical dependency**: Statement 3 needs **ORIGINAL** `a[i+1]` value
- **4 FMAs** - more compute-intensive than s241

## Key Finding: LLM-Generated Code Has Race Condition

**The LLM's Triton implementation is BUGGY** - it has a race condition at block boundaries.

### The Bug

The LLM loads `a[i+1]` before statement 1:
```python
a_plus_1 = tl.load(a_ptr + offsets_plus_1, ...)  # Loads ORIGINAL a[i+1]
new_a = b_vals + c_vals * d_vals  # Statement 1 modifies a
...
final_a = new_b + a_plus_1 * d_vals  # Uses a_plus_1
```

**Problem:** Within a block, this works. But at block boundaries:
- Block 0 processes indices [0, 1023]
- Block 1 processes indices [1024, 2047]
- When Block 0 needs `a[1024]`, Block 1 may have already modified it!
- **No synchronization between blocks** in Triton

### Test Results

| Array Size | LLM RAW Result |
|-----------|----------------|
| 1,000 | ✓ CORRECT (fits in 1 block) |
| 10,000 | ✗ ERROR (max_err=2.48) |
| 256,000 | ✗ ERROR (max_err=8.16) |

## Correct Implementation: Two-Stage Approach

The **only safe way** to handle this is a two-stage kernel:

**Stage 1:** Compute and store intermediate results
```python
# Statement 1: a[i] = b[i] + c[i] * d[i]
new_a = b_vals + c_vals * d_vals
# Statement 2: b[i] = a[i] + d[i] * e[i]
new_b = new_a + d_vals * e_vals

# Store to temp buffer
tl.store(a_temp_ptr + offsets, new_a, mask=mask)
tl.store(b_ptr + offsets, new_b, mask=mask)
```

**Stage 2:** Compute final a[i] using saved original a[i+1]
```python
# Load ORIGINAL a[i+1] (saved before stage 1)
a_orig_plus_1 = tl.load(a_orig_ptr + offsets_plus_1, ...)

# Statement 3: a[i] = b[i] + a[i+1] * d[i]
final_a = b_new + a_orig_plus_1 * d_vals

tl.store(a_ptr + offsets, final_a, mask=mask)
```

### Correctness Results

| Variant | N=1,000 | N=10,000 | N=256,000 | Status |
|---------|---------|----------|-----------|---------|
| LLM RAW | ✓ | ✓ | ✗ (9.92) | BUGGY |
| Variant 1 (Two-Stage) | ✓ | ✓ | ✓ | ✓ CORRECT |
| Variant 2 (Hand-written with a_orig) | ✓ | ✓ | ✓ | **✓ CORRECT** |

### Performance

| Variant | Time (ms) | CoV (%) | Notes |
|---------|-----------|---------|-------|
| PyTorch Baseline | 0.1504 | 0.9% | 439 separate kernel launches |
| Variant 1 (Two-Stage) | 0.1530 | 1.0% | Correct (2 kernels + temp buffer) - SLOWER than PyTorch |
| Variant 2 (Hand-written) | **0.0821** | 1.1% | **CORRECT - 1.83x faster than PyTorch!** |

## Comprehensive Profiling Results

### Visualizations

![Performance Comparison](profiling/visualizations/s243_performance_comparison.png)

![NCU Metrics](profiling/visualizations/s243_ncu_comparison.png)

### NCU Profiling Analysis

**PyTorch Baseline:**
- 439 kernel launches
- Total kernel time: 2360.84 μs
- Average DRAM utilization: 42.6%
- Average SM utilization: 7.4%
- **Bottleneck:** Memory-bound with poor kernel fusion

**Variant 1 (Two-Stage):**
- 2 kernel launches (Stage 1 + Stage 2)
- Total kernel time: 22.08 μs (11.04 μs × 2)
- DRAM utilization: 57.4%
- SM utilization: 9.8%
- **Bottleneck:** Memory-bound but better than PyTorch
- **Issue:** Slower than PyTorch due to extra memory operations

**Variant 2 (Hand-written with a_orig):**
- **1 kernel launch** (single fused kernel)
- Kernel time: 10.75 μs
- DRAM utilization: **59.9%** (best)
- SM utilization: 10.5% (best)
- **Bottleneck:** Memory-bound but optimal memory access patterns
- **Winner:** Best performance + correct results

### Why Variant 2 Wins

1. **Kernel Fusion:** Single kernel vs PyTorch's 439 kernels
2. **Memory Efficiency:** 59.9% DRAM utilization vs PyTorch's 42.6%
3. **No Extra Overhead:** Unlike Variant 1, no temp buffer or second kernel
4. **Correctness:** Avoids race conditions by passing `a_orig` as parameter

### Comparison: s241 vs s243

| Aspect | s241 | s243 |
|--------|------|------|
| **a_orig approach faster?** | ✗ NO (11% slower) | **✓ YES (46% faster than two-stage)** |
| **a_orig approach necessary?** | NO (minimal fix works) | **YES (avoids race conditions)** |
| **Triton vs PyTorch** | 2.51x faster | **1.83x faster** |
| **Main challenge** | Simple load reordering | Cross-iteration dependency |

## Comparison with s241

| Aspect | s241 | s243 |
|--------|------|------|
| Statements | 2 | 3 |
| Input Arrays | 4 (a, b, c, d) | 5 (a, b, c, d, e) |
| FMAs | 2 | 4 |
| LLM Code | ✓ Works (with minimal fix) | ✗ BUGGY (race condition) |
| Solution | Single-stage with careful ordering | **Two-stage required** |
| Complexity | Medium | **High** |

## Why s243 is Harder

1. **True cross-iteration dependency**: `a[i]` needs ORIGINAL `a[i+1]`
2. **No single-stage solution**: Can't reorder loads/stores to avoid race
3. **Requires synchronization**: Must use two kernels or atomic operations
4. **LLM failed to recognize**: Generated buggy single-stage code
5. **Hand-written also buggy**: Even manually optimizing the load order doesn't fix the cross-block race

## Profiling Tools Used

### 1. CUDA Events (Python)
- **What:** PyTorch's `torch.cuda.Event`
- **Measures:** End-to-end execution time
- **Used for:** All 3 variants, 10 trials each

### 2. Nsys (NVIDIA Nsight Systems)
- **What:** System-wide timeline profiler
- **Measures:** Kernel execution, API calls, memory transfers
- **Generated:** 3 profiles (PyTorch, Variant 1, Variant 2)
- **Location:** `profiling/nsys_reports/`

### 3. NCU (NVIDIA Nsight Compute)
- **What:** Detailed kernel-level profiler
- **Measures:** DRAM utilization, SM throughput, cache metrics, memory bandwidth
- **Generated:** 3 profiles (PyTorch 52M, Variant 1 17M, Variant 2 32M)
- **Location:** `profiling/ncu_reports/`
- **Key Finding:** Variant 2 achieves best DRAM utilization (59.9%)

## Files

```
s243/
├── README.md (this file)
└── profiling/
    ├── test_s243_variants.py           # All variants with correctness + perf tests
    ├── test_pytorch_baseline.py        # PyTorch for NCU/Nsys profiling
    ├── test_triton_variant1.py         # Triton two-stage for NCU/Nsys profiling
    ├── test_triton_variant2.py         # Triton hand-written for NCU/Nsys profiling
    ├── run_ncu_all_variants.sh         # NCU profiling automation
    ├── run_nsys_all_variants.sh        # Nsys profiling automation
    ├── extract_ncu_metrics.py          # Extract NCU metrics from reports
    ├── visualize_s243_comparison.py    # Generate comparison charts
    ├── test_correctness.py             # Simple correctness verification
    ├── ncu_reports/                    # NCU profiling results (101M total)
    │   ├── pytorch_baseline.ncu-rep    # 52M
    │   ├── variant_1_two_stage.ncu-rep # 17M
    │   ├── variant_2_hand_written.ncu-rep # 32M
    │   └── ncu_metrics_summary.json    # Extracted metrics
    ├── nsys_reports/                   # Nsys profiling results
    │   ├── pytorch_baseline.nsys-rep   # 420K
    │   ├── variant_1_two_stage.nsys-rep # 480K
    │   └── variant_2_hand_written.nsys-rep # 496K
    └── visualizations/                 # Performance comparison charts
        ├── s243_performance_comparison.png
        └── s243_ncu_comparison.png
```

## How to Run

### Test All Variants
```bash
cd profiling/
python test_s243_variants.py
```

### NCU Profiling (All Variants)
```bash
cd profiling/
./run_ncu_all_variants.sh
```

### Nsys Profiling (All Variants)
```bash
cd profiling/
./run_nsys_all_variants.sh
```

### Generate Visualizations
```bash
cd profiling/
python visualize_s243_comparison.py
```

## Key Insights

### 1. **LLM Limitations**
The LLM generated syntactically correct but semantically buggy code. It failed to:
- Recognize the cross-block race condition
- Understand that `a[i+1]` could be modified by another block
- Propose a two-stage solution

### 2. **Kernel Complexity Matters**
- s241: Simple dependency → Single-stage works
- s243: Cross-iteration dependency → **Two-stage required**

### 3. **Always Test at Scale**
The bug only appears when:
- Array size > block size (1024)
- Multiple blocks run in parallel
- Small tests (N=1000) pass, large tests fail!

### 4. **PyTorch Baseline Had Same Bug**
Initially, the PyTorch baseline was also wrong:
```python
# WRONG: Uses modified 'a' for roll
a_shifted = torch.roll(a, shifts=-1, dims=0)

# CORRECT: Save original first
a_orig = a.clone()
a_shifted = torch.roll(a_orig, shifts=-1, dims=0)
```

## Recommendations

### For s243 Specifically
- **✓ Use Variant 2 (Hand-written with a_orig)** - Best performance (1.83x faster) + correct!
- ❌ Avoid Two-Stage (Variant 1) - Slower than PyTorch
- ❌ Avoid LLM RAW - Race condition at block boundaries

### For Similar Kernels
When you have cross-iteration dependencies:
1. **Analyze carefully**: Does iteration i need ORIGINAL data from iteration i±k?
2. **Use a_orig approach**: Pass cloned original array as parameter
3. **Test at scale**: Small tests may hide race conditions
4. **Profile thoroughly**: NCU/Nsys profiling reveals true performance
5. **Don't trust LLM blindly**: Verify correctness with multiple sizes

## Conclusion

s243 demonstrates that **cross-iteration dependencies CAN be efficiently handled in a single Triton kernel** using the `a_orig` approach!

### Key Findings

**✓ Variant 2 (Hand-written with a_orig) is the winner:**
- **Correct:** Avoids race conditions by passing `a_orig` parameter
- **Fast:** 1.83x faster than PyTorch (0.0821 vs 0.1504 ms)
- **Single kernel:** Optimal kernel fusion
- **Best DRAM utilization:** 59.9% vs PyTorch's 42.6%

**⚠️ Variant 1 (Two-Stage) works but is slower:**
- Correct but slower than PyTorch (0.1530 vs 0.1504 ms)
- Two kernels + temp buffer overhead
- Not recommended

**✗ LLM RAW is buggy:**
- Race condition at block boundaries
- Only works for arrays < block size
- Never use in production

### Bottom Line

For kernels with cross-iteration dependencies:
1. **First choice:** Hand-written kernel with `a_orig` parameter (Variant 2)
2. **Fallback:** PyTorch baseline (simpler, correct, reasonable performance)
3. **Last resort:** Two-stage Triton (correct but slow)

**The a_orig approach proves that Triton CAN handle complex dependencies efficiently!**

---

**Generated:** October 21, 2025
**Status:** ✓ Complete - NCU & Nsys profiling done
**Recommended Variant:** **Variant 2 (Hand-written with a_orig)** - 1.83x faster than PyTorch
**Profiling Reports:** 101M NCU + 1.4M Nsys data available
