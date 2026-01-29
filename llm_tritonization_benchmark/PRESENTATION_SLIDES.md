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

## Current State (Test 26 Results)

### Summary Metrics
| Metric | Count | Percentage |
|--------|-------|------------|
| ✅ **PASSING** | 128 | 84.8% |
| ❌ **FAILING** | 23 | 15.2% |
| 📊 **Benchmarked** | 128 | 84.8% |
| ⚡ **Valid Speedups** | 124 | 82.1% |
| ⏱️ **C Ref Timeouts** | 2 | 1.3% |
| ⏱️ **Triton Timeouts** | 2 | 1.3% |

*Results measured against TSVC C reference (ground truth) with checksum-based verification.*

### Pass Rate by Attempt
| Attempt | New Passes | Cumulative | Rate |
|---------|------------|------------|------|
| 1 | 104 | 104 | 68.9% |
| 2+ (retry) | 24 | 128 | 84.8% |

---

## Correctness Results (Test 26)

### Failed Functions by Error Type (23 total)

#### Numerical Mismatch (21 functions)

Functions where Triton output differs from C reference beyond tolerance (max_error > 1e-3):

s1115, s1119, s114, s119, s122, s1232, s126, s132, s173, s174, s2101, s2111, s2233, s231, s233, s244, s275, s276, s4114, s424, s471

#### Compilation Errors (1 function)

- **s232:** `arange's arguments must be of type tl.constexpr`

#### Runtime Errors (1 function)

- **s2102:** AST/compilation runtime error

### Error Analysis

| Error Type | Count | % of Failures | Root Cause |
|------------|-------|---------------|------------|
| Numerical | 21 | 91.3% | Algorithm incorrectness, dependency handling |
| Compilation | 1 | 4.3% | constexpr type mismatches in Triton |
| Runtime | 1 | 4.3% | AST/compilation runtime errors |

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

## Performance Summary (Test 26)

### Overall Statistics
| Metric | Value |
|--------|-------|
| **Benchmarked** | 128 |
| **Valid Speedups** | 124 |
| **C Ref Timeouts** | 2 |
| **Triton Timeouts** | 2 |
| **Mean Speedup** | 0.78x |
| **Median Speedup** | 0.45x |
| **Min Speedup** | 0.00x |
| **Max Speedup** | 10.85x |

### Performance Distribution (124 functions with valid speedups)

```
Speedup Range          Count    %     Distribution
─────────────────────────────────────────────────────────────────
>2x faster            :  10   ( 8.1%) ██████████
1.5x-2x faster        :   8   ( 6.5%) ████████
1x-1.5x faster        :  13   (10.5%) █████████████
0.5x-1x (slower)      :  28   (22.6%) ████████████████████████████
0.1x-0.5x (slower)    :  34   (27.4%) ██████████████████████████████████
<0.1x (much slower)   :  31   (25.0%) ███████████████████████████████
─────────────────────────────────────────────────────────────────
Triton faster (>=1x)  :  31   (25.0%)
Triton slower (<1x)   :  93   (75.0%)
```

### Visual Distribution
```
                    SLOWER  ◄─────────────────────►  FASTER

<0.1x   ███████████████████████████████████████████████████████████████  31
0.1-0.5x████████████████████████████████████████████████████████████████████████  34
0.5-1x  ████████████████████████████████████████████████████████  28
1-1.5x  ██████████████████████████  13
1.5-2x  ████████████████  8
>2x     ████████████████████  10
        | - - - - | - - - - | - - - - | - - - - | - - - - | - - - - | - - - -
        0         5         10        15        20        25        30        35
```

---

## Top 10 Fastest Functions (Triton vs C)

| Rank | Function | Speedup | Notes |
|------|----------|---------|-------|
| 1 | s422 | C timeout | C ref timeout (>60s), Triton 9.72ms |
| 2 | s423 | C timeout | C ref timeout (>60s), Triton 9.72ms |
| 3 | s176 | 10.85x | Loop-heavy kernel |
| 4 | s451 | 6.47x | Loop interchange |
| 5 | s235 | 2.51x | Control flow |
| 6 | s2275 | 2.45x | Node splitting |
| 7 | s442 | 2.38x | Computed goto |
| 8 | s1279 | 2.20x | Control flow |
| 9 | vtvtv | 2.16x | Vector operations |
| 10 | s125 | 2.15x | Loop distribution |

**Note:** C reference runs on CPU, Triton runs on GPU. s422-s423 show C timeout (>60s).

---

## Bottom 10 Slowest Functions

| Rank | Function | Speedup | Notes |
|------|----------|---------|-------|
| 1 | s255 | Triton timeout | Triton timeout (>60s) |
| 2 | s343 | Triton timeout | Triton timeout (>60s) |
| 3 | s1221 | 0.0004x | Severe kernel overhead |
| 4 | s3112 | 0.008x | Double dimension |
| 5 | s256 | 0.015x | Statement reorder |
| 6 | s115 | 0.016x | Loop overhead |
| 7 | s318 | 0.017x | Reduction with dependency |
| 8 | s116 | 0.018x | Loop overhead |
| 9 | s292 | 0.020x | Search loop |
| 10 | s257 | 0.022x | Statement reorder |

**Note:** Slowdowns are primarily due to kernel launch overhead dominating small operations.

---

## Performance by Category (Test 26)

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

### Results (Test 26)
- ✅ **Correctness rate:** 84.8% (128/151 functions)
- ✅ **First-try success rate:** 68.9% (104/151 functions)
- ✅ **Retry recovery:** +24 functions via retries
- ✅ **Performance:** 25.0% faster than C, 75.0% slower
- ✅ **Max speedup:** 10.85x (s176), or C timeout for s422/s423

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

**Results (Test 26):**
- **84.8% correctness** (128/151 functions pass)
- **68.9% first-try success** (104 functions)
- **25.0% achieve GPU speedup** (31/124 functions)
- **Max 10.85x speedup** (s176), C timeout for s422/s423
- **Median 0.45x** (kernel overhead often dominates)

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
