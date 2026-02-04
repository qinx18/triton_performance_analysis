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

### 2. **TSVC Function Database + C Code Parser**
```python
# Database stores minimal info (name + source code)
TSVC_FUNCTIONS = {
    "s421": {
        "name": "s421",
        "loop_code": "for (int i = 0; i < n; i++) { ... }"
    },
    # ... 150 more functions
}

# Properties inferred at runtime via c_code_parser.py:
# - arrays: extracted from array accesses (a[i] patterns)
# - has_offset: detected from index patterns ([i+10], [i-1])
# - has_reduction: detected from accumulation (sum +=, x = x + ...)
# - has_conditional: detected from if statements
# - has_2d_arrays: detected from [i][j] patterns
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

## Current State (Test 29 Results)

### Summary Metrics
| Metric | Count | Percentage |
|--------|-------|------------|
| **PASSING** | 150 | 99.3% |
| **FAILING** | 1 | 0.7% |
| **Benchmarked** | 151 | 100% |
| **Valid Speedups** | 148 | 98.0% |
| **C Ref Timeouts** | 3 | 2.0% |
| **Triton Timeouts** | 0 | 0.0% |

*Results measured against TSVC C reference (ground truth) with checksum-based verification.*

### Pass Rate by Attempt
| Attempt | New Passes | Cumulative | Rate |
|---------|------------|------------|------|
| 1 | 120 | 120 | 79.5% |
| 2-5 | 18 | 138 | 91.4% |
| 6-10 | 12 | 150 | 99.3% |

### Comparison: Test 28 (Initial) → Test 29
| Metric | Test 28 Initial | Test 29 | Change |
|--------|-----------------|---------|--------|
| Pass Rate | 100% (151/151) | 99.3% (150/151) | -1 function |
| First-try Success | 82.8% (125) | 79.5% (120) | -5 functions |
| Mean Speedup | 1.95x | 2.50x | +28% ✓ |
| Max Speedup | 246.41x | 220.28x | -10.6% |
| Functions <0.1x | 25 | 3 | -22 ✓ |
| Functions >=1x | 48 (32.4%) | 53 (35.8%) | +5 ✓ |

---

## Correctness Results (Test 29)

### 150/151 Functions Pass (99.3%)

Test 29 achieved **99.3% correctness** — 150 of 151 functions pass correctness tests.

**1 Failed Function:**

| Function | Error Type | Root Cause |
|----------|------------|------------|
| **s424** | compilation | Sequential loop pattern + C ref timeout |

Note: Pass rate varies by run due to LLM non-determinism.

### Failure Pattern Analysis (Test 29)

**s424** has a C reference timeout (-1 speedup) and consistently fails with
compilation errors. The sequential access pattern is difficult to optimize.

### Error Summary (Test 29)

| Error Type | Count | Root Cause |
|------------|-------|------------|
| Compilation | 1 | Triton limitations (s424) |

---

## Key Correctness Insights

### 1. **99%+ Pass Rate Achieved**
- Test 29: 150/151 (99.3%)
- Test 28 Initial: 151/151 (100%) - but with poor performance
- 1 persistent failure: s424 (C ref timeout)

### 2. **Static Analysis Improvements**
- Prefix sum detection with tl.cumsum() strategy
- Strided prefix sum with per-stream parallelization
- Loop distribution with verified_safe analysis
- Statement reordering for RAW dependencies
- **Goto pattern conversion** for control flow (s277: 0.057x → 2.05x)

### 3. **Retry Strategy Effectiveness**
- 5+5 reset strategy: 29 functions recovered via retries (118→147)
- Some patterns (s421, s424) consistently fail due to Triton limitations
- s277 now passes with goto conversion analysis

### 4. **LLM Handles Complex Patterns Well**
- 2D loops with dependencies
- Atomic operations for scatter patterns
- Statement reordering for RAW dependencies
- Scalar expansion for temporary variables
- Conditional parallelization
- Stream compaction with cumsum
- Goto-based control flow (via conversion)

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

## Performance Summary (Test 29)

### Overall Statistics
| Metric | Value |
|--------|-------|
| **Benchmarked** | 149 |
| **Valid Speedups** | 146 |
| **C Ref Timeouts** | 3 (s422, s423, s424) |
| **Triton Timeouts** | 0 |
| **Mean Speedup** | 2.50x |
| **Min Speedup** | 0.060x (s323) |
| **Max Speedup** | 220.28x (s176) |

### Performance Distribution (148 functions with valid speedups)

```
Speedup Range          Count    %     Distribution
─────────────────────────────────────────────────────────────────
>=2x faster           :  24   (16.2%) ████████████████████████
1.5x-2x faster        :  10   ( 6.8%) ██████████
1x-1.5x faster        :  19   (12.8%) ███████████████████
0.5x-1x (slower)      :  53   (35.8%) █████████████████████████████████████████████████████
0.1x-0.5x (slower)    :  39   (26.4%) ███████████████████████████████████████
<0.1x (much slower)   :   3   ( 2.0%) ███
─────────────────────────────────────────────────────────────────
Triton faster (>=1x)  :  53   (35.8%)
Triton slower (<1x)   :  95   (64.2%)
```

### Visual Distribution Comparison: Test 28 (Initial) → Test 29

```
                         Test 28 (148)                  Test 29 (148)                    Change
                    ─────────────────────────────────────────────────────────────────────
<0.1x (slowest)     █████████████████████████  25     ███  3                        -22  ✓
0.1x-0.5x           ███████████████████████████████████████████████████  51   ███████████████████████████████████████  39        -12  ✓
0.5x-1x             ████████████████████████████  28   █████████████████████████████████████████████████████  53        +25  ✓
1x-1.5x             ████████████  12                  ███████████████████  19             +7  ✓
1.5x-2x             ██████████████  14                ██████████  10                 -4
>=2x (fastest)      ██████████████████████  22        ████████████████████████  24         +2  ✓
                    ─────────────────────────────────────────────────────────────────────
Triton >= 1x:       48 (32.4%)                        53 (35.8%)                       +5  ✓

Key improvements from analysis modules:
- s277: 0.057x → 2.05x (goto conversion analysis)
- s1221: 0.0004x → 0.15x (strided prefix sum)
- s256: 0.037x → 4.49x (j-sequential recurrence)
- s3251: 0.049x → 0.83x (loop distribution)
- s119: Now passes with wavefront parallelism (0.81x)
- s323: Now passes after loop distribution bug fix (0.11x)
- s421: Now passes after retry with constexpr fix (0.56x)
- 22 functions moved out of <0.1x tier
```

---

## Top 10 Fastest Functions (Triton vs C)

| Rank | Function | Speedup | Notes |
|------|----------|---------|-------|
| 1 | s176 | 220.28x | Loop-heavy kernel (C=129.4ms, T=0.6ms) |
| 2 | s451 | 5.96x | Loop interchange |
| 3 | s2111 | 5.41x | Double dimension |
| 4 | s256 | 4.49x | j-sequential recurrence |
| 5 | s233 | 4.21x | Control flow |
| 6 | s2102 | 3.39x | Statement reorder |
| 7 | s231 | 3.38x | Control flow |
| 8 | s343 | 3.26x | Recurrence |
| 9 | s126 | 3.22x | Loop distribution |
| 10 | s2233 | 3.05x | Node splitting |

**Note:** C reference runs on CPU, Triton runs on GPU. s422/s423/s424 show C timeout (>60s).

---

## Bottom 10 Slowest Functions

| Rank | Function | Speedup | Notes |
|------|----------|---------|-------|
| 1 | s323 | 0.060x | Nested loop (failed, benchmarked last attempt) |
| 2 | s321 | 0.077x | Recurrence pattern |
| 3 | s257 | 0.079x | Sequential pattern |
| 4 | s322 | 0.088x | Recurrence |
| 5 | s123 | 0.100x | Sequential |
| 6 | s332 | 0.121x | Simple loop |
| 7 | s316 | 0.133x | Simple loop |
| 8 | s311 | 0.139x | Simple loop |
| 9 | s331 | 0.139x | Simple loop |
| 10 | s1221 | 0.153x | Strided prefix sum |

**Note:** Slowdowns are primarily due to kernel launch overhead dominating small operations. s422/s423/s424 had C reference timeouts (>60s) with Triton completing in ~10ms.

### Significant Improvements from Test 28 (Initial) → Test 29

| Function | Test 28 Initial | Test 29 | Improvement | Root Cause |
|----------|-----------------|---------|-------------|------------|
| s1221 | 0.0004x | 0.15x | **340x** | Strided prefix sum analysis |
| s256 | 0.037x | 4.49x | **121x** | j-sequential recurrence optimization |
| s277 | 0.057x | 2.05x | **36x** | Goto conversion analysis |
| s222 | 0.026x | 0.95x | **37x** | Loop distribution optimization |
| s3251 | 0.049x | 0.83x | **17x** | Loop distribution (verified parallel) |
| s251 | 0.049x | 0.79x | **16x** | Loop overhead reduction |
| s261 | 0.076x | 1.24x | **16x** | Statement reorder optimization |
| s253 | 0.090x | 1.01x | **11x** | Scalar expansion fix |
| s3112 | 0.045x | 0.39x | **9x** | Prefix sum with tl.cumsum() |
| s2111 | 0.092x | 5.41x | **59x** | Double dimension optimization |

---

## Performance by Category (Test 28 (Current))

### Performance Tiers Observed

| Tier | Categories | Avg Speedup | Notes |
|------|------------|-------------|-------|
| 🚀 High (>1.5x) | Loop interchange, Control flow | 1.5-5.6x | High parallelism benefit |
| ⚡ Moderate (1-1.5x) | Vector ops, Statement reorder | 1.0-1.5x | Balanced overhead/benefit |
| 🐌 Low (<0.5x) | Reductions, Simple loops | 0.02-0.5x | Kernel overhead dominates |

### Key Performance Patterns

**What achieves speedup:**
- Loop-heavy kernels (s176: 10.85x)
- Loop interchange patterns (s451: 6.47x)
- Complex control flow (s273, s274: ~2x)
- Conditional vector operations (vif: 2.05x)
- Computed goto patterns (s442: 2.38x)
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

### Results (Test 29)
- **Correctness rate:** 99.3% (150/151 functions)
- **First-try success rate:** 79.5% (120/151 functions)
- **Retry recovery:** +30 functions via retries
- **Performance:** 35.8% faster than C, 64.2% slower
- **Max speedup:** 220.28x (s176)
- **Mean speedup:** 2.50x
- **Key improvement:** 22 functions moved out of <0.1x tier vs Test 28

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

**Results (Test 29):**
- **99.3% correctness** (150/151 functions pass)
- **79.5% first-try success** (120 functions)
- **35.8% achieve GPU speedup** (53/148 functions)
- **Max 220.28x speedup** (s176)
- **Mean 2.50x speedup**
- **22 functions improved** from <0.1x tier vs initial

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
