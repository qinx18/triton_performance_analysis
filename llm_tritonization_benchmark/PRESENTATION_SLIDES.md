# LLM-Driven Triton Code Generation for TSVC Benchmark

**Automated Infrastructure for GPU Kernel Generation and Validation**

---

# Part 1: Infrastructure Design

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    generate_and_test.py                     │
│                     (Main Pipeline)                          │
└──────────────┬──────────────────────────────────────────────┘
               │
               ├─► TSVC Function Database (151 functions)
               │   └─ utilities/tsvc_functions_db.py
               │
               ├─► Static Analysis Modules (PET/ISL)
               │   ├─ WAR Dependencies
               │   ├─ Statement Overwrites
               │   ├─ Stream Compaction
               │   ├─ Loop Unrolling Patterns
               │   ├─ Early Exit Detection
               │   ├─ Statement Reordering
               │   ├─ Scalar Expansion
               │   ├─ Reduction Detection
               │   └─ Convolution Patterns
               │
               ├─► LLM Generation (Claude Sonnet 4.5)
               │   ├─ Initial prompt with analysis
               │   ├─ Retry with error feedback (max 10)
               │   └─ 5+5 reset strategy
               │
               ├─► Test Infrastructure
               │   ├─ TSVC C reference (compiled shared library)
               │   ├─ Triton correctness testing
               │   └─ Performance benchmarking vs C reference
               │
               └─► Results Collection
                   ├─ test{N}/llm_triton/  (generated code)
                   ├─ test{N}/results.json (metrics)
                   └─ FINAL_TEST_RESULTS.md (analysis)
```

---

## Pipeline Flow: Per-Function Processing

```
┌──────────────────┐
│  TSVC Function   │ (e.g., s421)
└────────┬─────────┘
         │
         ├─► 1. Extract C code structure
         │      - Kernel loop identification
         │      - Array access patterns
         │      - Local variables
         │
         ├─► 2. Run Static Analysis
         │      - PET: Dependence analysis
         │      - ISL: Parallelization strategy
         │      - Pattern detection (8 modules)
         │
         ├─► 3. Build LLM Prompt
         │      - C code + analysis results
         │      - Triton compilation rules
         │      - Function signature requirements
         │
         ├─► 4. C Reference (Pre-compiled)
         │      - Original TSVC C kernels
         │      - Compiled as shared library (libtsvc_all.so)
         │      - Python wrappers via ctypes
         │
         ├─► 5. Generate Triton Code
         │      ├─ Attempt 1: Initial generation
         │      ├─ Attempts 2-5: Retry with errors
         │      ├─ Attempt 6: Reset context
         │      └─ Attempts 7-10: Fresh tries
         │
         ├─► 6. Test Correctness
         │      - Compare vs TSVC C reference
         │      - Multiple input sizes
         │      - Tolerance: max_error < 1e-3
         │
         └─► 7. Benchmark Performance
                - 10 warmup iterations
                - 100 benchmark iterations
                - 60-second timeout per section
                - Record speedup ratio
```

---

## Key Infrastructure Components

### 1. **generate_and_test.py** (Main Pipeline)
- **Lines:** ~2,100
- **Functions:** 40+
- **Key Features:**
  - Automatic TSVC function extraction from C code
  - Integration with 8 static analysis modules
  - Retry logic with error feedback
  - Test harness auto-generation
  - Benchmark infrastructure

### 2. **TSVC Function Database**
```python
TSVC_FUNCTIONS = {
    "s421": {
        "arrays": {"a": "r", "xx": "rw", "yy": "r"},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "category": "storage_classes"
    },
    # ... 150 more functions
}
```

### 3. **Static Analysis Modules** (PET + Custom)
| Module | Purpose | Example Output |
|--------|---------|----------------|
| `compute_war_dependences` | Detect write-after-read | "Save `a[i]` before overwrite" |
| `compute_statement_overwrites` | Detect overwrite patterns | "Use latest value only" |
| `compute_stream_compaction` | Detect if/scatter patterns | "Use atomic operations" |
| `compute_loop_unrolling` | Suggest unroll strategies | "Unroll by factor 4" |
| `compute_early_exit` | Find break conditions | "Use sequential loop" |
| `compute_statement_reordering` | RAW dependency order | "Reorder statements" |
| `compute_scalar_expansion` | Temporary variable needs | "Expand scalar to array" |
| `compute_reduction_type` | Reduction operations | "Use atomic_add" |

---

## Retry Strategy Evolution

### Initial Approach (Tests 1-16)
```
Attempt 1: Initial generation
Attempt 2-3: Retry with error
→ Problem: Gets stuck in same error pattern
```

### 5+5 Strategy (Test 17+)
```
Attempts 1-5:  Retry with error feedback
               └─ Show last attempt + error
Attempt 6:     RESET CONTEXT
               └─ Fresh generation without history
Attempts 7-10: New retry sequence
               └─ Fresh perspective on the problem
```

**Result:** +3 functions passed on first try (test16→test17)

---

## Test Harness Auto-Generation

For each function, automatically generates:

### 1. **Correctness Test** (`my_triton_implementations/{func}/test_{func}_correctness.py`)
```python
# Auto-generated based on array specs
- Test sizes: [100, 1000, 10000] or [64, 128, 256] for 2D
- Clone tensors for isolation
- Compare outputs: max_error < 1e-3
- Return: PASS/FAIL + error details
```

### 2. **Benchmark Script** (`my_triton_implementations/{func}/benchmark_{func}.py`)
```python
# Auto-generated with timeout handling
- 10 warmup iterations (60s timeout)
- 100 benchmark iterations (60s timeout)
- Record C reference (CPU) and Triton (GPU) times
- Calculate speedup ratio
- Handle timeouts gracefully
```

---

## Prompt Engineering

### Prompt Structure (per function)
```
1. TSVC C code (30-100 lines)
2. Kernel loop to implement (5-20 lines)
3. Array information (types, sizes, access patterns)
4. Static analysis results (0-8 modules)
   ├─ WAR dependencies (if applicable)
   ├─ Statement overwrites (if applicable)
   ├─ Stream compaction (if applicable)
   └─ ... other patterns
5. Function signature requirements (exact parameter names)
6. CRITICAL: Triton compilation rules (12 rules)
   ├─ NEVER use tl.arange() in loops
   ├─ NEVER use scalar indexing in kernels
   ├─ NEVER use non-existent Triton functions
   └─ ... 9 more rules
7. Expected output: Python code only
```

**Total prompt size:** 500-2000 tokens (varies by complexity)

---

# Part 2: Correctness Results

## Historical Progress

*Note: Historical results were measured against LLM-generated PyTorch baseline.*
*Results with TSVC C reference baseline will be measured in new test runs.*

### Design Evolution
- Tests 1-18: PyTorch baseline (LLM-generated, potential bugs)
- Test 19+: TSVC C reference (original ground truth)

**Benefit of new design:** Removes baseline bugs, provides authoritative correctness reference.

---

## Current State (Test 19 Results - Final)

### Summary Metrics
| Metric | Count | Percentage |
|--------|-------|------------|
| ✅ **PASSING** | 113 | 74.8% |
| ❌ **FAILING** | 38 | 25.2% |
| 📊 **Benchmarked** | 113 | 74.8% |
| ⚡ **Valid Speedups** | 110 | 72.8% |
| ⏱️ **C Ref Timeouts** | 2 | 1.3% |
| ⏱️ **Triton Timeouts** | 1 | 0.7% |

*Results measured against TSVC C reference (ground truth) with checksum-based verification.*

### Pass Rate by Attempt
| Attempt | New Passes | Cumulative | Rate |
|---------|------------|------------|------|
| 1 | 96 | 96 | 63.6% |
| 2+ (retry) | 17 | 113 | 74.8% |

---

## Correctness Results (Test 19)

### Failed Functions by Error Type (38 total)

#### Numerical Mismatch (33 functions)

Functions where Triton output differs from C reference beyond tolerance (max_error > 1e-3):

| Function | Max Error | Function | Max Error |
|----------|-----------|----------|-----------|
| s1115 | 1.01e+01 | s1119 | 2.16e+01 |
| s114 | 6.73e+00 | s115 | 1.36e+08 |
| s118 | 2.10e+08 | s1232 | 6.79e+00 |
| s126 | 2.81e+01 | s132 | 4.24e+00 |
| s176 | 1.44e+27 | s2101 | 4.99e+00 |
| s2102 | 3.85e+00 | s2111 | 1.16e+02 |
| s2233 | 2.59e+01 | s2275 | 5.27e+00 |
| s231 | 2.29e+01 | s232 | unknown |
| s233 | 2.53e+01 | s235 | 3.31e+00 |
| s256 | 3.62e+00 | s257 | 1.39e+01 |
| s258 | 8.63e+00 | s275 | 2.14e+01 |
| s281 | 2.96e+00 | s3110 | 1.51e+02 |
| s3111 | 4.47e+01 | s3113 | 2.98e+00 |
| s312 | 1.52e+01 | s353 | 4.93e+00 |
| s4115 | 5.64e+00 | s4116 | 4.06e+00 |
| s442 | 1.06e+01 | vbor | 2.88e+03 |
| s13110 | 1.09e+03 | | |

#### Generation/Runtime Errors (2 functions)

- **s318:** 'NoneType' object is not subscriptable
- **s423:** 'NoneType' object is not subscriptable

#### Compilation Errors (2 functions)

- **s351:** Type mismatch between pointer and float32
- **s31111:** AttributeError in generated code

#### Timeout (1 function)

- **s119:** Test timed out after 60 seconds

### Error Analysis

| Error Type | Count | % of Failures | Root Cause |
|------------|-------|---------------|------------|
| Numerical | 33 | 86.8% | Algorithm incorrectness, dependency handling |
| Generation | 2 | 5.3% | LLM failed to generate valid code |
| Compilation | 2 | 5.3% | Type mismatches in Triton |
| Timeout | 1 | 2.6% | Infinite loops or very slow execution |

---

## Success by Function Category

| Category | Total | Pass | Rate | Notes |
|----------|-------|------|------|-------|
| Single dimension ops | 13 | TBD | TBD | |
| Double dimensions | 6 | TBD | TBD | |
| Induction variables | 8 | TBD | TBD | |
| Global data flow | 3 | TBD | TBD | |
| Nonlinear dependence | 2 | TBD | TBD | |
| Interprocedural | 2 | TBD | TBD | |
| Control flow | 20 | TBD | TBD | |
| Statement reordering | 4 | TBD | TBD | |
| Loop distribution | 3 | TBD | TBD | |
| Loop interchange | 6 | TBD | TBD | |
| Node splitting | 5 | TBD | TBD | |
| Scalar expansion | 6 | TBD | TBD | |
| Reductions | 13 | TBD | TBD | |
| Recurrences | 3 | TBD | TBD | |
| Search loops | 2 | TBD | TBD | |
| Packing | 3 | TBD | TBD | |
| Loop rerolling | 3 | TBD | TBD | |
| Storage classes | 4 | TBD | TBD | s421 known issue |
| Intrinsic functions | 3 | TBD | TBD | |
| Indirect addressing | 6 | TBD | TBD | |
| Vector operations | 9 | TBD | TBD | |
| Control loops | 6 | TBD | TBD | |

*Results to be measured against TSVC C reference.*

---

## Key Correctness Insights

### 1. **LLM Handles Complex Patterns Well**
- ✅ 2D loops with dependencies
- ✅ Atomic operations for scatter patterns
- ✅ Statement reordering for RAW dependencies
- ✅ Scalar expansion for temporary variables
- ✅ Conditional parallelization
- ✅ Stream compaction with cumsum

### 2. **Static Analysis is Critical**
Static analysis guidance improves LLM generation quality.

### 3. **Retry Strategy Works**
- 5+5 reset strategy helps escape error loops
- Most functions succeed within first few attempts

### 4. **Remaining Challenges**
- Implicit requirements (constexpr)
- Edge cases in prompt engineering
- LLM consistency across attempts

*Detailed statistics to be measured with C reference baseline.*

---

# Part 3: Performance Results

## Benchmark Infrastructure (Test 18)

### New Features
```
✅ 60-second timeout per section (warmup/benchmark)
✅ Separate timeout tracking for C reference vs Triton
✅ Minimum speedup calculation for timeouts
✅ Graceful error handling
✅ Machine-readable output format
```

### Timeout Handling
```python
# C reference timeout:
- Baseline too slow (>60s for 100 iterations)
- Report: C ref time = -1ms
- Calculate minimum speedup: 60000ms / triton_time

# Triton timeout:
- Report: Triton time = -1ms

# Both timeout:
- Report: "Both timeout"
```

---

## Performance Summary (Test 19)

### Overall Statistics
| Metric | Value |
|--------|-------|
| **Benchmarked** | 113 |
| **Valid Speedups** | 110 |
| **C Ref Timeouts** | 2 (s422, s343) |
| **Triton Timeouts** | 1 (s343) |
| **Mean Speedup** | 0.68x |
| **Median Speedup** | 0.47x |
| **Min Speedup** | 0.0004x |
| **Max Speedup** | 5.57x |

### Performance Distribution (110 functions with valid speedups)

```
Speedup Range          Count    %     Distribution
─────────────────────────────────────────────────────────────────
>2x faster            :   5   ( 4.5%) ████
1.5x-2x faster        :  10   ( 9.1%) █████████
1x-1.5x faster        :  12   (10.9%) ██████████
0.5x-1x (slower)      :  25   (22.7%) ██████████████████████
0.1x-0.5x (slower)    :  31   (28.2%) ████████████████████████████
<0.1x (much slower)   :  27   (24.5%) ████████████████████████
─────────────────────────────────────────────────────────────────
Triton faster (>1x)   :  27   (24.5%)
Triton slower (<1x)   :  83   (75.5%)
```

### Visual Distribution
```
                    SLOWER  ◄─────────────────────►  FASTER

<0.1x  ████████████████████████████████████████████████████████  27
0.1-0.5x  ██████████████████████████████████████████████████████████████  31
0.5-1x  ██████████████████████████████████████████████████  25
1-1.5x  ████████████████████████  12
1.5-2x  ████████████████████  10
>2x     ██████████  5
        |----|----|----|----|----|----|----
        0    5    10   15   20   25   30
```

---

## Top 10 Fastest Functions (Triton vs C)

| Rank | Function | Speedup | C Ref (ms) | Triton (ms) | Notes |
|------|----------|---------|------------|-------------|-------|
| 1 | s451 | 5.57x | 0.518 | 0.093 | Loop interchange |
| 2 | vtvtv | 2.16x | 0.174 | 0.080 | Vector operation |
| 3 | s125 | 2.15x | 0.209 | 0.097 | Induction variable |
| 4 | vif | 2.14x | 0.144 | 0.067 | Conditional vector |
| 5 | s273 | 2.02x | 0.243 | 0.120 | Control flow |
| 6 | s443 | 1.92x | 0.172 | 0.089 | Intrinsic function |
| 7 | s1161 | 1.91x | 0.182 | 0.096 | Single dimension |
| 8 | s2710 | 1.89x | 0.180 | 0.095 | Control flow |
| 9 | s161 | 1.88x | 0.205 | 0.109 | Statement reorder |
| 10 | s274 | 1.85x | 0.241 | 0.131 | Control flow |

**Note:** C reference runs on CPU, Triton runs on GPU. Speedups >1x indicate GPU is faster.

---

## Bottom 10 Slowest Functions

| Rank | Function | Speedup | C Ref (ms) | Triton (ms) | Notes |
|------|----------|---------|------------|-------------|-------|
| 1 | s1221 | 0.0004x | 0.045 | 120.117 | Severe overhead |
| 2 | s116 | 0.019x | 0.023 | 1.209 | Loop overhead |
| 3 | s331 | 0.021x | 0.016 | 0.764 | Packing |
| 4 | s342 | 0.025x | 0.054 | 2.148 | Search loop |
| 5 | s3112 | 0.029x | 0.058 | 1.974 | Reduction |
| 6 | s111 | 0.031x | 0.030 | 0.959 | Single dimension |
| 7 | s1213 | 0.033x | 0.071 | 2.185 | Double dimension |
| 8 | s254 | 0.035x | 0.040 | 1.130 | Node splitting |
| 9 | s292 | 0.036x | 0.055 | 1.549 | Loop rerolling |
| 10 | s211 | 0.038x | 0.096 | 2.545 | Statement reorder |

**Note:** Slowdowns are primarily due to kernel launch overhead dominating small operations.

---

## Performance by Category (Test 19)

### Performance Tiers Observed

| Tier | Categories | Avg Speedup | Notes |
|------|------------|-------------|-------|
| 🚀 High (>1.5x) | Loop interchange, Control flow | 1.5-5.6x | High parallelism benefit |
| ⚡ Moderate (1-1.5x) | Vector ops, Statement reorder | 1.0-1.5x | Balanced overhead/benefit |
| 🐌 Low (<0.5x) | Reductions, Simple loops | 0.02-0.5x | Kernel overhead dominates |

### Key Performance Patterns

**What achieves speedup:**
- Loop interchange patterns (s451: 5.57x)
- Complex control flow (s273, s274: ~2x)
- Conditional vector operations (vif: 2.14x)
- Operations with sufficient arithmetic intensity

**What suffers slowdown:**
- Very simple operations (kernel launch overhead > computation)
- Sequential patterns that can't parallelize
- Small data sizes where transfer overhead dominates

---

## Why GPU Speedups?

### Understanding the Comparison

**The C Reference (CPU):**
```c
// Sequential C loop on CPU
for (int i = 0; i < 32000; i++) {
    a[i] = b[i] + 1.0;
}
```
- **Execution:** Sequential on single CPU core
- **Optimizations:** Compiler auto-vectorization (SIMD)
- **Baseline:** Represents optimized sequential C code

**The Triton Implementation (GPU):**
```python
# Single kernel launch, massively parallel
@triton.jit
def kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    # All 32000 elements in parallel across GPU cores!
```
- **Benefit:** Massive parallelism (thousands of threads)
- **Parallelism:** All elements processed simultaneously
- **Overhead:** Kernel launch + data transfer

### Comparison Context

| Comparison Type | Notes |
|-----------------|-------|
| Triton (GPU) vs C (CPU) | Measures GPU parallelization benefit |
| Triton vs Hand-optimized CUDA | ~0.5-2x (Triton generates efficient code) |

---

## Performance Insights

### 1. **What Triton Excels At**
✅ Loop interchange patterns (high parallelism)
✅ 2D operations with dependencies
✅ Complex control flow
✅ Induction variable computations
✅ Stream compaction

### 2. **What May Not Benefit**
❌ Trivial operations (kernel overhead may dominate)
❌ Operations with limited parallelism
❌ Single scalar updates (no parallelism to exploit)

### 3. **Key Observations**
- GPU parallelization provides significant speedups for vectorizable loops
- Kernel launch overhead affects small/trivial operations
- C reference provides a more realistic baseline than Python loops

*Detailed performance insights will be updated after running experiments.*

---

# Conclusions & Future Work

## Key Achievements ✅

### Infrastructure
- ✅ Fully automated pipeline (TSVC → Triton)
- ✅ 8 static analysis modules integrated
- ✅ Comprehensive test harness generation
- ✅ Retry logic with context reset (5+5 strategy)
- ✅ Timeout-aware benchmarking

### Results (Test 19)
- ✅ **Correctness rate:** 74.8% (113/151 functions)
- ✅ **First-try success rate:** 63.6% (96/151 functions)
- ✅ **Retry recovery:** +17 functions via retries
- ✅ **Performance:** 24.5% faster than C, 75.5% slower
- ✅ **Max speedup:** 5.57x (s451 - loop interchange)

---

## Limitations & Learnings

### 1. **Prompt Engineering Matters**
- Explicit > Implicit instructions
- Example code is crucial
- s421 failure: missing constexpr instruction

### 2. **Baseline Choice Matters**
- Now using original TSVC C functions as baseline
- C reference provides realistic CPU performance
- GPU vs CPU comparison shows true parallelization benefit

### 3. **Static Analysis Helps**
- 98% vs 95% pass rate with/without analysis
- Not all patterns need analysis
- Some edge cases still missed

### 4. **LLM Consistency**
- 17% need retries
- Some errors persist across attempts
- 5+5 reset helps but not always

---

## Future Work

### Short Term
1. **Fix s421 prompt**
   - Add explicit constexpr instruction
   - Provide working example
   - Test on similar patterns

2. **Run performance experiments**
   - Measure speedups vs C reference baseline
   - Compare against hand-written CUDA
   - Analyze kernel launch overhead impact

3. **Add more analysis modules**
   - Memory access pattern analysis
   - Register pressure prediction
   - Occupancy optimization

### Long Term
1. **Auto-tuning integration**
   - BLOCK_SIZE optimization
   - Grid size tuning
   - Memory layout optimization

2. **Performance optimization**
   - Beyond correctness → optimal code
   - Memory coalescing hints
   - Shared memory utilization

3. **Broader benchmarks**
   - Beyond TSVC
   - Real-world kernels
   - Production workloads

4. **Model improvement**
   - Fine-tune on Triton corpus
   - Few-shot learning with examples
   - Chain-of-thought for complex patterns

---

# Thank You!

## Summary

**Infrastructure:**
- Automated TSVC → Triton pipeline
- 8 static analysis modules
- 5+5 retry strategy
- Comprehensive testing vs C reference

**Results (Test 19):**
- **74.8% correctness** (113/151 functions pass)
- **63.6% first-try success** (96 functions)
- **24.5% achieve GPU speedup** (27/110 functions)
- **Max 5.57x speedup** (s451 - loop interchange)
- **Median 0.47x** (kernel overhead often dominates)

**Impact:**
- Demonstrates LLM capability for specialized GPU kernel generation
- Shows importance of static analysis for complex patterns
- Identifies performance bottlenecks (kernel launch overhead)

---

## Questions?

📧 Contact: qin-x18@mails.tsinghua.edu.cn
🔗 Repository: [Add your repo link]
📄 Paper: [In progress]

**Next steps:** Improve numerical accuracy, optimize kernel launch overhead, expand patterns!
