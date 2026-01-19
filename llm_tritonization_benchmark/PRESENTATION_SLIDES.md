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

## Current State (To Be Measured)

### Summary Metrics
| Metric | Count | Percentage |
|--------|-------|------------|
| ✅ **PASSING** | TBD | TBD |
| ❌ **FAILING** | TBD | TBD |
| 📊 **Benchmarked** | TBD | TBD |
| ⚡ **Valid Speedups** | TBD | TBD |
| ⏱️ **C Ref Timeouts** | TBD | TBD |

*Note: Results will be measured against original TSVC C reference functions (ground truth).*
*Previous results were measured against LLM-generated PyTorch baseline which may have contained bugs.*

### Pass Rate by Attempt
| Attempt | New Passes | Cumulative | Rate |
|---------|------------|------------|------|
| TBD | TBD | TBD | TBD |

---

## Correctness Results (To Be Measured)

*Correctness results will be measured against original TSVC C reference functions.*

### Key Changes from Previous Design
- **Old baseline:** LLM-generated PyTorch code (potential bugs)
- **New baseline:** Original TSVC C functions (ground truth)
- **Benefit:** Removes potential baseline bugs, provides authoritative reference

### Known Issue: s421

**Error:** `ValueError: arange's arguments must be of type tl.constexpr`

**Root Cause:** LLM consistently generates incorrect kernel signature
```python
# Generated (WRONG):
@triton.jit
def s421_kernel(xx_ptr, yy_ptr, a_ptr, n):
    BLOCK_SIZE = 256                    # ❌ Regular variable
    offsets = tl.arange(0, BLOCK_SIZE)  # ❌ Compilation error

# Expected (CORRECT):
@triton.jit
def s421_kernel(xx_ptr, yy_ptr, a_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)  # ✅ Works!
```

**Recommendation:** Add explicit constexpr instruction to prompt

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

## Performance Summary

### Overall Statistics
| Metric | Value |
|--------|-------|
| **Benchmarked** | TBD |
| **Valid Speedups** | TBD |
| **C Ref Timeouts** | TBD |
| **Triton Timeouts** | TBD |
| **Average Speedup** | TBD |
| **Median Speedup** | TBD |

*Note: Performance statistics to be measured after running experiments with C reference baseline.*

### Performance Distribution
```
Functions faster than baseline:  TBD
Functions slower than baseline:  TBD
Functions with C ref timeout:    TBD
```

---

## Performance Results (To Be Measured)

Performance comparisons will be measured against the original TSVC C reference functions.

### Expected Comparison
| Comparison | Notes |
|------------|-------|
| Triton (GPU) vs C Reference (CPU) | Measures GPU acceleration benefit |
| Single kernel launch vs sequential C | Shows parallelization advantage |

*Detailed performance results will be populated after running experiments.*

---

## Top 10 Measured Speedups

| Function | Speedup | Triton (ms) | C Ref (ms) | Category |
|----------|---------|-------------|------------|----------|
| TBD | TBD | TBD | TBD | TBD |

*Speedup measurements will be populated after running experiments with C reference baseline.*

**Note:** C reference runs on CPU, Triton runs on GPU. Speedups reflect GPU parallelization benefits over sequential CPU execution.

---

## Performance by Category

*Performance by category will be measured after running experiments with C reference baseline.*

### Expected Performance Tiers
| Tier | Expected Categories | Notes |
|------|---------------------|-------|
| 🚀 High Speedup | Loop interchange, 2D operations | High parallelism potential |
| ⚡ Good Speedup | Most vectorizable loops | Standard GPU acceleration |
| 🐌 Limited Speedup | Simple operations, trivial loops | Kernel launch overhead may dominate |

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
- ✅ Retry logic with context reset
- ✅ Timeout-aware benchmarking

### Results (To Be Measured)
- ⏳ **Correctness rate** (vs TSVC C reference)
- ⏳ **First-try success rate**
- ⏳ **Category pass rates**
- ⏳ **Performance vs C reference**

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
- Comprehensive testing

**Results (To Be Measured):**
- Correctness vs TSVC C reference
- First-try success rate
- Performance speedups (GPU vs CPU)

**Impact:**
- Demonstrates LLM capability for specialized code generation
- Shows value of static analysis integration
- Provides framework for future GPU kernel automation

---

## Questions?

📧 Contact: qin-x18@mails.tsinghua.edu.cn
🔗 Repository: [Add your repo link]
📄 Paper: [In progress]

**Next steps:** Fix s421, improve baselines, expand to more benchmarks!
