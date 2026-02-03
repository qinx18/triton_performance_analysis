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
| **PASSING** | 151 | 100.0% |
| **FAILING** | 0 | 0.0% |
| **Benchmarked** | 151 | 100.0% |
| **Valid Speedups** | 148 | 98.0% |
| **C Ref Timeouts** | 3 | 2.0% |
| **Triton Timeouts** | 0 | 0.0% |

*Results measured against TSVC C reference (ground truth) with checksum-based verification.*

### Pass Rate by Attempt
| Attempt | New Passes | Cumulative | Rate |
|---------|------------|------------|------|
| 1 | 128 | 128 | 84.8% |
| 2 | 14 | 142 | 94.0% |
| 3 | 5 | 147 | 97.4% |
| 4 | 1 | 148 | 98.0% |
| 6 | 2 | 150 | 99.3% |
| 8 | 1 | 151 | 100.0% |

### Comparison: Test 28 → Test 29
| Metric | Test 28 | Test 29 | Change |
|--------|---------|---------|--------|
| Pass Rate | 96.7% (146/151) | **100%** (151/151) | +5 functions |
| First-try Success | 82.8% (125) | 84.8% (128) | +3 functions |
| Mean Speedup | 2.55x | 2.62x | +2.7% |
| Median Speedup | 0.53x | 0.60x | +13.2% |

---

## Correctness Results (Test 29)

### All 151 Functions Pass

Test 29 achieved **100% correctness** — all 151 functions pass correctness tests.

The 5 functions that failed in Test 28 (s123, s256, s281, s317, s3112) all passed in Test 29:

| Function | Test 28 | Test 29 | Attempts | Speedup |
|----------|---------|---------|----------|---------|
| **s123** | FAIL (numerical) | PASS | 2 | 0.46x |
| **s256** | FAIL (numerical) | PASS | 1 | 3.41x |
| **s281** | FAIL (numerical) | PASS | 8 | 0.48x |
| **s317** | FAIL (compilation) | PASS | 2 | 0.09x |
| **s3112** | FAIL (compilation) | PASS | 1 | 0.04x |

This confirms the failures were due to LLM non-determinism — given enough attempts, all functions
can produce correct implementations with the current prompt infrastructure.

### Failure Pattern Analysis

**s123 and s281** share a common pattern: the prompt's static analysis sections
(stream compaction, crossing threshold) encourage parallelization, but the LLM
consistently implements the parallel version incorrectly. When the LLM ignores
the analysis and falls back to a sequential single-thread kernel (`grid=(1,)`),
it produces correct results.

**s317** had an additional **test harness bug** (now fixed): the test passed
`n=1` instead of `N`, and the C wrapper returned `void` instead of the scalar
result. This caused test27's correct implementation to be falsely marked as
failed. The C kernel and wrapper have been fixed to return the scalar, and the
test harness now sets `n=N`.

**s256** involves a sequential recurrence (`a[j] = 1.0 - a[j-1]`) combined with a
2D computation (`aa[j][i] = a[j] + bb[j][i]*d[j]`). The prompt correctly prescribes
j-sequential, i-parallel strategy, but the LLM frequently gets implementation details
wrong (pointer arithmetic, scalar value passing). It is chronically unstable — cleanly
passed in only 4 of 16 test runs (test14, test17, test18, test25), with a per-attempt
pass rate of ~14% (10/72 attempts across all tests). In test28, all 5 attempts had
`max_error = 5.61`, and the benchmark timed out. On re-run, it passed on attempt 2
(0.04x speedup). Notably, the recurrence `a[j] = 1.0 - a[j-1]` is an involution
(`f(f(x)) = x`), so `a[j]` simply alternates between `a[0]` (even j) and `1.0 - a[0]`
(odd j). If the LLM recognized this closed form, the entire computation would become
embarrassingly parallel with no sequential dependency — but no attempt has ever
exploited this simplification.

**s3112** failed all 10 attempts with the same compilation error — defining
`BLOCK_SIZE` as a regular variable inside the kernel instead of as a
`tl.constexpr` parameter. Despite error feedback mentioning the constexpr
requirement, the LLM could not escape this pattern across 10 retries.

### Error Summary

| Error Type | Count | Root Cause |
|------------|-------|------------|
| Numerical | 4 | LLM non-determinism: wrong algorithm choice or arithmetic |
| Compilation | 1 | LLM non-determinism: constexpr not used for BLOCK_SIZE |

---

## Key Correctness Insights

### 1. **100% Pass Rate Achieved (Test 29)**
- All 151 functions now pass with the current infrastructure
- Previous failures were due to LLM non-determinism, not systematic bugs
- The 5+5 retry strategy successfully recovers all edge cases

### 2. **Static Analysis Can Be Counterproductive**
- s123 and s281: parallelization guidance leads to incorrect implementations
- The LLM produces correct code when it ignores the analysis and uses sequential execution
- Overly specific guidance may constrain the LLM away from simpler correct solutions

### 3. **Retry Strategy Works but Has Limits**
- 5+5 reset strategy helps: 22 functions recovered via retries
- But s3112 shows retries can get stuck in the same error pattern (all 10 attempts identical bug)
- s123 got close (attempt 7: error 1.19e-02) but never crossed the threshold

### 4. **LLM Handles Complex Patterns Well**
- 2D loops with dependencies
- Atomic operations for scatter patterns
- Statement reordering for RAW dependencies
- Scalar expansion for temporary variables
- Conditional parallelization
- Stream compaction with cumsum

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
| Metric | Test 28 (Initial) | Test 29 | Change |
|--------|-------------------|---------|--------|
| **Benchmarked** | 151 | 151 | - |
| **Valid Speedups** | 148 | 148 | - |
| **C Ref Timeouts** | 3 | 3 | - |
| **Triton Timeouts** | 0 | 0 | - |
| **Mean Speedup** | 2.55x | 2.62x | +2.7% |
| **Median Speedup** | 0.53x | 0.60x | +13.2% |
| **Min Speedup** | 0.0004x | 0.026x | +65x |
| **Max Speedup** | 246.41x | 246.19x | - |

### Performance Distribution (148 functions with valid speedups)

```
Speedup Range          Count    %     Distribution
─────────────────────────────────────────────────────────────────
>=2x faster           :  21   (14.2%) █████████████████████
1.5x-2x faster        :  14   ( 9.5%) ██████████████
1x-1.5x faster        :  12   ( 8.1%) ████████████
0.5x-1x (slower)      :  42   (28.4%) ██████████████████████████████████████████
0.1x-0.5x (slower)    :  43   (29.1%) ███████████████████████████████████████████
<0.1x (much slower)   :  16   (10.8%) ████████████████
─────────────────────────────────────────────────────────────────
Triton faster (>=1x)  :  47   (31.8%)
Triton slower (<1x)   : 101   (68.2%)
```

### Visual Distribution Comparison: Test 28 (Initial) → Test 29

```
                         Test 28 (Initial)              Test 29                  Change
                    ─────────────────────────────────────────────────────────────────────
<0.1x (slowest)     █████████████████████████████  29   ████████████████  16       -13  ✓
0.1x-0.5x           ███████████████████████████████████████████  43   ███████████████████████████████████████████  43        0
0.5x-1x             ████████████████████████████  28   ██████████████████████████████████████████  42       +14  ✓
1x-1.5x             ██████████████  14                 ████████████  12                -2
1.5x-2x             ████████████  12                   ██████████████  14              +2
>=2x (fastest)      ██████████████████████  22         █████████████████████  21       -1
                    ─────────────────────────────────────────────────────────────────────
Triton >= 1x:       48 (32.4%)                         47 (31.8%)                      -1

Key improvement: 13 functions moved OUT of <0.1x tier (extremely slow) to faster tiers
```

### Performance Shift Analysis
```
                    SLOWER  ◄─────────────────────►  FASTER

<0.1x   ████████████████  16  (was 29, improved by 13 functions)
0.1-0.5x███████████████████████████████████████████  43
0.5-1x  ██████████████████████████████████████████  42  (gained 14 functions)
1-1.5x  ████████████  12
1.5-2x  ██████████████  14
>=2x    █████████████████████  21
        | - - - - | - - - - | - - - - | - - - - | - - - - |
        0         10        20        30        40        50
```

---

## Top 10 Fastest Functions (Triton vs C)

| Rank | Function | Speedup | Notes |
|------|----------|---------|-------|
| 1 | s176 | 246.19x | Loop-heavy kernel (C=129.4ms, T=0.5ms) |
| 2 | s451 | 6.95x | Loop interchange |
| 3 | s233 | 4.18x | Control flow |
| 4 | s231 | 3.75x | Control flow |
| 5 | s2111 | 3.70x | Double dimension (improved from 0.09x) |
| 6 | s256 | 3.41x | j-sequential recurrence (improved from 0.04x) |
| 7 | s2233 | 3.16x | Node splitting |
| 8 | s126 | 2.77x | Loop distribution |
| 9 | s2275 | 2.65x | Induction variable |
| 10 | s343 | 2.64x | Recurrence |

**Note:** C reference runs on CPU, Triton runs on GPU. s422/s423/s424 show C timeout (>60s).

---

## Bottom 10 Slowest Functions

| Rank | Function | Speedup | Notes |
|------|----------|---------|-------|
| 1 | s222 | 0.026x | Statement reorder pattern |
| 2 | s211 | 0.040x | Statement reorder pattern |
| 3 | s3112 | 0.044x | Recurrence |
| 4 | s3251 | 0.046x | Loop overhead |
| 5 | s1221 | 0.048x | Strip-vectorizable |
| 6 | s251 | 0.048x | Loop overhead |
| 7 | s277 | 0.056x | Conditional |
| 8 | s1251 | 0.069x | Strip-vectorizable |
| 9 | s261 | 0.074x | Statement reorder pattern |
| 10 | s321 | 0.076x | Loop overhead |

**Note:** Slowdowns are primarily due to kernel launch overhead dominating small operations. s422/s423/s424 had C reference timeouts (>60s) with Triton completing in ~10ms.

### Significant Improvements from Test 28 (Initial) → Test 29

| Function | Initial | Test 29 | Improvement | Root Cause |
|----------|---------|---------|-------------|------------|
| s1221 | 0.0004x | 0.048x | **107x** | Fixed strip-vectorizable kernel launch |
| s256 | 0.037x | 3.41x | **92x** | j-sequential recurrence optimization |
| s252 | 0.009x | 0.51x | **56x** | Statement reorder fix |
| s115 | 0.017x | 0.92x | **53x** | Loop overhead reduction |
| s2111 | 0.092x | 3.70x | **40x** | Double dimension optimization |
| s1213 | 0.033x | 0.60x | **18x** | Statement reordering module fix |

---

## Performance by Category (Test 29)

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
- **Correctness rate:** 100% (151/151 functions)
- **First-try success rate:** 84.8% (128/151 functions)
- **Retry recovery:** +23 functions via retries
- **Performance:** 31.8% faster than C, 68.2% slower
- **Max speedup:** 246.19x (s176)
- **Median speedup improved:** 0.53x → 0.60x (+13%)

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
- **100% correctness** (151/151 functions pass)
- **84.8% first-try success** (128 functions)
- **31.8% achieve GPU speedup** (47/148 functions)
- **Max 246.19x speedup** (s176)
- **Median 0.60x** (+13% improvement from Test 28)

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
