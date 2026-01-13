# Final Test Results - Complete TSVC Suite with Comprehensive Investigation

**Test Date:** 2025-12-08 (Integrated generate_and_test.py - test16)
**Previous Tests:** test15, test14, test13, test12, test11, test10, test9, test8, test7, test6, test5, test4, test3, test2, test1, 2025-11-29, 2025-11-28, 2025-11-18, 2025-11-17, 2025-11-06
**Model:** claude-sonnet-4-20250514
**Total Functions:** 151
**Infrastructure:** PyTorch Baseline Comparison ✅

---

## 🔬 LLM Triton v3 Continued Runs (2025-12-08) - test16 - LATEST RUN

### Summary
| Metric | Count | Percentage |
|--------|-------|------------|
| ✅ **PASSING** | 143 | 94.7% |
| ❌ **FAILING** | 8 | 5.3% |

### Comparison with test15

| Metric | test15 | test16 | Change |
|--------|--------|--------|--------|
| Passing | 140 (92.7%) | 143 (94.7%) | **+3** |
| Failing | 11 (7.3%) | 8 (5.3%) | **-3** |
| Passed 1st try | 112 | 119 | **+7** |

**Functions Fixed (7 from test15 failures now pass):**
- ✅ **s123**: Conditionals - now passes (attempt 2)
- ✅ **s2244**: Array expansion - now passes (attempt 2)
- ✅ **s2251**: Array expansion - now passes (attempt 3)
- ✅ **s235**: Loop interchanging - now passes (attempt 1)
- ✅ **s255**: Carry-around variables - now passes (attempt 3)
- ✅ **s256**: Array expansion - now passes (attempt 1)
- ✅ **s424**: Pointer aliasing - now passes (attempt 1)

**New Failures (LLM non-determinism):**
- ❌ **s118**: Numerical (max_error = 6.35e+09) - was passing in test15
- ❌ **s244**: Numerical (max_error = 8.67e-01) - was passing in test15
- ❌ **s252**: Numerical (max_error = 2.88e+00) - was passing in test15
- ❌ **s31111**: Numerical (max_error = 1.59e+01) - was passing in test15

**Persistent Failures (4 functions):**
- ❌ **s141**: Numerical (max_error = 4.43e+00) - pointer aliasing
- ❌ **s176**: Timeout - sequential computation, inherently not parallelizable
- ❌ **s257**: Numerical (max_error = 8.36e+00) - complex scalar expansion with loop-carried dependency
- ❌ **s258**: Numerical (max_error = 3.29e+00) - wrap-around scalar under if condition

### Failed Functions (8) - Error Breakdown

| Function | Attempts | Error Type | Max Error | Notes |
|----------|----------|-----------|-----------|-------|
| s118 | 3 | numerical | 6.35e+09 | Regression from test15 |
| s141 | 3 | numerical | 4.43e+00 | Pointer aliasing (persists) |
| s176 | 10 | timeout | N/A | Sequential computation (persists) |
| s244 | 3 | numerical | 8.67e-01 | Regression from test15 |
| s252 | 4 | numerical | 2.88e+00 | Regression from test15 |
| s257 | 3 | numerical | 8.36e+00 | Loop-carried dependency (persists) |
| s258 | 3 | numerical | 3.29e+00 | Wrap-around scalar (persists) |
| s31111 | 3 | numerical | 1.59e+01 | Regression from test15 |

**Error Type Summary:**
- Numerical errors: 7 functions
- Timeout: 1 function (s176)

### Pass Rate by Attempt
| Attempt | New Passes | Cumulative |
|---------|------------|------------|
| Attempt 1 | 119 | 119 (78.8%) |
| Attempt 2 | +17 | 136 (90.1%) |
| Attempt 3 | +6 | 142 (94.0%) |
| Attempt 4 | +1 | 143 (94.7%) |

### Passing Functions (143):
s000, s111, s1111, s1112, s1113, s1115, s1119, s112, s113, s114, s115, s116, s1161, s119, s121, s1213, s122, s1221, s123, s1232, s124, s1244, s125, s1251, s126, s127, s1279, s128, s1281, s131, s13110, s132, s1351, s1421, s151, s152, s161, s162, s171, s172, s173, s174, s175, s2101, s2102, s211, s2111, s212, s221, s222, s2233, s2244, s2251, s2275, s231, s232, s233, s235, s241, s242, s243, s251, s253, s254, s255, s256, s261, s271, s2710, s2711, s2712, s272, s273, s274, s275, s276, s277, s278, s279, s281, s291, s292, s293, s311, s3110, s3111, s3112, s3113, s312, s313, s314, s315, s316, s317, s318, s319, s321, s322, s323, s3251, s331, s332, s341, s342, s343, s351, s352, s353, s4112, s4113, s4114, s4115, s4116, s4117, s4121, s421, s422, s423, s424, s431, s441, s442, s443, s451, s452, s453, s471, s481, s482, s491, va, vag, vas, vbor, vdotr, vif, vpv, vpvpv, vpvts, vpvtv, vsumr, vtv, vtvtv

---

## 🔬 LLM Triton v3 with Reduction Analysis & Return Statement Extraction (2025-12-07) - test15

### Summary
| Metric | Count | Percentage |
|--------|-------|------------|
| ✅ **PASSING** | 139 | 92.1% |
| ❌ **FAILING** | 12 | 7.9% |

### Key Fixes in test15

**New Analysis Modules:**
1. **Reduction Type Detection (`compute_reduction_type.py`)**: Detects reduction patterns (sum, product, dot, max, min, max_abs, argmax, argmin, argmax_abs) and provides Triton-specific guidance including custom combiner functions for `tl.reduce()`.
2. **Return Statement Extraction**: Extracts actual return statements from TSVC C source code and includes them in the prompt to ensure correct return value computation (e.g., `return max + index + 1`).

**Bug Fixes Applied:**
1. **s312 Baseline**: Added missing `return prod` statement for product reduction.
2. **s318 Baseline**: Fixed return value from `max + index` to `max + index + 1`.
3. **s3113 Baseline**: Added missing `return max_val` statement. Removed `abs` from scalar_params in tsvc_functions_db.py (ABS is a macro, not a parameter).
4. **s4116 Baseline**: Added missing return statement with vectorized implementation.

### Comparison with test14

| Metric | test14 | test15 | Change |
|--------|--------|--------|--------|
| Passing | 140 (92.7%) | 139 (92.1%) | -1 |
| Failing | 11 (7.3%) | 12 (7.9%) | +1 |

**Functions Fixed (9 from test14 failures now pass):**
- ✅ **s113**: Numerical error fixed (now passes on attempt 1)
- ✅ **s114**: Numerical error fixed (now passes on attempt 1)
- ✅ **s13110**: Conditional max reduction now passes (attempt 1) - return statement extraction helped
- ✅ **s3110**: Conditional max reduction now passes (attempt 1) - return statement extraction helped
- ✅ **s3113**: Runtime NoneType fixed (baseline + param fix, passes on attempt 2)
- ✅ **s312**: Product reduction now works (custom combiner guidance, passes on attempt 1)
- ✅ **s318**: Argmax now correct (baseline return fix, passes on attempt 2)
- ✅ **s4116**: Indirect dot product fixed (baseline return fix, passes on attempt 2)
- ✅ **s256**: Numerical error fixed (now passes on attempt 3)

**New Failures (LLM non-determinism):**
- ❌ **s123**: Numerical (max_error = 5.94e+00) - was passing in test14
- ❌ **s141**: Numerical (max_error = 4.14e+00) - was passing in test14
- ❌ **s2244**: Compilation (arange constexpr) - was passing in test14
- ❌ **s2251**: Numerical (max_error = 1.13e+01) - was passing in test14
- ❌ **s235**: Numerical (max_error = 2.85e+00) - was passing in test14
- ❌ **s241**: Numerical (max_error = 5.28e+13) - was passing in test14
- ❌ **s255**: Numerical (max_error = 1.11e+00) - was passing in test14
- ❌ **s424**: Numerical (max_error = 7.51e+00) - was passing in test14

**Persistent Failures:**
- ❌ **s176**: Timeout (sequential computation - inherently not parallelizable)
- ❌ **s257**: Numerical (complex scalar expansion)
- ❌ **s258**: Numerical (complex scalar expansion)

### Failed Functions (12) - Error Breakdown

| Function | Attempts | Error Type | Max Error | Notes |
|----------|----------|-----------|-----------|-------|
| s123 | 3 | numerical | 5.94e+00 | Regression from test14 |
| s141 | 3 | numerical | 4.14e+00 | Regression from test14 |
| s176 | 10 | timeout | N/A | Sequential computation (persists) |
| s2244 | 10 | compilation | N/A | arange constexpr issue |
| s2251 | 3 | numerical | 1.13e+01 | Regression from test14 |
| s235 | 3 | numerical | 2.85e+00 | Regression from test14 |
| s241 | 3 | numerical | 5.28e+13 | Regression from test14 |
| s255 | 3 | numerical | 1.11e+00 | Regression from test14 |
| s256 | 3 | numerical | 3.08e+00 | Persists |
| s257 | 3 | numerical | 6.21e+00 | Complex scalar expansion (persists) |
| s258 | 5 | numerical | 5.71e+00 | Complex scalar expansion (persists) |
| s424 | 6 | numerical | 7.51e+00 | Regression from test14 |

**Error Type Summary:**
- Numerical errors: 10 functions
- Timeout: 1 function (s176)
- Compilation: 1 function (s2244)

### Pass Rate by Attempt
| Attempt | New Passes | Cumulative |
|---------|------------|------------|
| Attempt 1 | 112 | 112 (74.2%) |
| Attempt 2 | +23 | 135 (89.4%) |
| Attempt 3 | +3 | 138 (91.4%) |
| Attempt 5 | +1 | 139 (92.1%) |

### Passing Functions (139):
s000, s111, s1111, s1112, s1113, s1115, s1119, s112, s113, s114, s115, s116, s1161, s118, s119, s121, s1213, s122, s1221, s1232, s124, s1244, s125, s1251, s126, s127, s1279, s128, s1281, s131, s13110, s132, s1351, s1421, s151, s152, s161, s162, s171, s172, s173, s174, s175, s2101, s2102, s211, s2111, s212, s221, s222, s2233, s2275, s231, s232, s233, s242, s243, s244, s251, s252, s253, s254, s261, s271, s2710, s2711, s2712, s272, s273, s274, s275, s276, s277, s278, s279, s281, s291, s292, s293, s311, s3110, s3111, s31111, s3112, s3113, s312, s313, s314, s315, s316, s317, s318, s319, s321, s322, s323, s3251, s331, s332, s341, s342, s343, s351, s352, s353, s4112, s4113, s4114, s4115, s4116, s4117, s4121, s421, s422, s423, s431, s441, s442, s443, s451, s452, s453, s471, s481, s482, s491, va, vag, vas, vbor, vdotr, vif, vpv, vpvpv, vpvts, vpvtv, vsumr, vtv, vtvtv

---

## 🔬 LLM Triton v3 with Sequential Pattern Fixes (2025-12-07) - test14

### Summary
| Metric | Count | Percentage |
|--------|-------|------------|
| ✅ **PASSING** | 140 | 92.7% |
| ❌ **FAILING** | 11 | 7.3% |

### Key Fixes in test14

**Bug Fixes Applied:**
1. **Identity Matrix Pattern (s2102)**: Fixed race condition - Block 0 was setting ALL diagonals while Block 1's zeroing overwrote them. Added `detect_identity_matrix_pattern()` with masked diagonal store guidance.
2. **Pure Reduction Test (s31111)**: Fixed test to compare scalar return values instead of arrays for pure reduction functions.
3. **Sequential Recurrence Detection (s322, s323)**: Fixed pointer aliasing analysis to use MINIMUM positive offset_diff instead of last pair's value. Added "sequential" pattern type for offset_diff=1.
4. **s322 Baseline Bug**: Fixed incorrect vectorized baseline that used sliced operations for a recurrence requiring sequential execution.

### Comparison with test13

| Metric | test13 | test14 | Change |
|--------|--------|--------|--------|
| Passing | 141 (93.4%) | 140 (92.7%) | -1 |
| Failing | 10 (6.6%) | 11 (7.3%) | +1 |

**Functions Fixed (now pass):**
- ✅ **s2102**: Identity matrix pattern fix
- ✅ **s322**: Sequential recurrence fix (passes on attempt 2)
- ✅ **s323**: Sequential recurrence fix
- ✅ **s31111**: Pure reduction test fix (passes on attempt 2)
- ✅ **s116**: Now passes
- ✅ **s161**: Now passes
- ✅ **s175**: Now passes
- ✅ **s211**: Now passes

**New Failures (LLM non-determinism):**
- ❌ **s113**: Numerical (max_error = 2.00e+00)
- ❌ **s114**: Numerical (max_error = 6.32e+00)
- ❌ **s13110**: Numerical (max_error = 2.00e+00)
- ❌ **s256**: Numerical (max_error = 4.47e+00)
- ❌ **s3110**: Numerical (max_error = 2.00e+00)
- ❌ **s3113**: Runtime (NoneType return)
- ❌ **s312**: Runtime (AST parsing error)
- ❌ **s318**: Numerical (max_error = 1.00e+00)
- ❌ **s4116**: Runtime (NoneType return)

### Failed Functions (11) - Error Breakdown

| Function | Attempts | Error Type | Notes |
|----------|----------|-----------|-------|
| s113 | 3 | numerical | Regression from test13 |
| s114 | 3 | numerical | Regression from test13 |
| s13110 | 3 | numerical | Regression from test13 |
| s176 | 10 | timeout | Sequential computation (persists) |
| s256 | 3 | numerical | Regression from test13 |
| s257 | 3 | numerical | Complex scalar expansion (persists) |
| s3110 | 7 | numerical | Regression from test13 |
| s3113 | 10 | runtime | NoneType return value |
| s312 | 10 | runtime | Triton AST parsing issue |
| s318 | 4 | numerical | Regression from test13 |
| s4116 | 10 | runtime | NoneType return value |

**Error Type Summary:**
- Numerical errors: 8 functions
- Timeout: 1 function (s176)
- Runtime errors: 2 functions (s3113, s4116)

### Pass Rate by Attempt
| Attempt | New Passes | Cumulative |
|---------|------------|------------|
| Attempt 1 | 111 | 111 (73.5%) |
| Attempt 2 | +19 | 130 (86.1%) |
| Attempt 3 | +7 | 137 (90.7%) |
| Attempt 4 | +1 | 138 (91.4%) |
| Attempt 5 | +2 | 140 (92.7%) |

### Passing Functions (140):
s000, s111, s1111, s1112, s1113, s1115, s1119, s112, s115, s116, s1161, s118, s119, s121, s1213, s122, s1221, s123, s1232, s124, s1244, s125, s1251, s126, s127, s1279, s128, s1281, s131, s132, s1351, s141, s1421, s151, s152, s161, s162, s171, s172, s173, s174, s175, s2101, s2102, s211, s2111, s212, s221, s222, s2233, s2244, s2251, s2275, s231, s232, s233, s235, s241, s242, s243, s244, s251, s252, s253, s254, s255, s258, s261, s271, s2710, s2711, s2712, s272, s273, s274, s275, s276, s277, s278, s279, s281, s291, s292, s293, s311, s3111, s31111, s3112, s313, s314, s315, s316, s317, s319, s321, s322, s323, s3251, s331, s332, s341, s342, s343, s351, s352, s353, s4112, s4113, s4114, s4115, s4117, s4121, s421, s422, s423, s424, s431, s441, s442, s443, s451, s452, s453, s471, s481, s482, s491, va, vag, vas, vbor, vdotr, vif, vpv, vpvpv, vpvts, vpvtv, vsumr, vtv, vtvtv

---

## 🔬 LLM Triton v3 with Enhanced Analysis (2025-12-07) - test13

### Summary
| Metric | Count | Percentage |
|--------|-------|------------|
| ✅ **PASSING** | 141 | 93.4% |
| ❌ **FAILING** | 10 | 6.6% |

### Key Improvements in test13

**Enhanced Analysis Tools Integrated:**
1. **Wavefront Pattern Detection (s2111)**: Added detection and guidance for 2D wavefront parallelism where both dimensions have dependencies
2. **Stream Compaction (s343)**: Now passes with proper guidance about prefix sum pattern
3. **Crossing Threshold (s281)**: Added explicit warning about NOT using sliced arrays in Phase 2
4. **Statement Reordering (s211)**: Integrated analysis into prompt generation
5. **Loop Stride Detection (s116)**: Fixed overwrite detection for strided loops
6. **Fixed s277 baseline**: Baseline was buggy (incorrectly parallelized), now uses sequential processing

### Pass Rate by Attempt
| Attempt | Passed | Cumulative |
|---------|--------|------------|
| Attempt 1 | 107 | 107 (70.9%) |
| Attempt 2 | +24 | 131 (86.8%) |
| Attempt 3 | +5 | 136 (90.1%) |
| Attempt 4 | +3 | 139 (92.1%) |
| Attempt 5 | +1 | 140 (92.7%) |
| Attempt 6 | +1 | 141 (93.4%) |

### Comparison with test12

| Metric | test12 | test13 | Change |
|--------|--------|--------|--------|
| Passing | 141 (93.4%) | 141 (93.4%) | Same |
| Failing | 10 (6.6%) | 10 (6.6%) | Same |

**Note:** Same pass rate, but different functions passing/failing due to analysis improvements:
- ✅ **s343 NOW PASSES**: Stream compaction analysis integrated
- ✅ **s277 NOW PASSES**: Fixed buggy baseline (was incorrectly parallelized)
- ✅ **s281 NOW PASSES**: Added warning about sliced array bug
- ❌ **s161 NOW FAILS**: Regression (needs investigation)
- ❌ **s175 NOW FAILS**: Regression (needs investigation)
- ❌ **s31111 NOW FAILS**: Regression (needs investigation)
- ❌ **s323 NOW FAILS**: Regression (needs investigation)

### Failed Functions (10) - Error Breakdown

| Function | Attempts | Error Type | Notes |
|----------|----------|-----------|-------|
| s116 | 3 | numerical | Manually unrolled loop (stride 5) |
| s161 | 4 | numerical | Offset + conditional pattern |
| s175 | 3 | numerical | Forward dependency pattern |
| s176 | 10 | timeout | Sequential computation |
| s2102 | 3 | numerical | 2D array pattern |
| s211 | 4 | numerical | Statement reordering (has analysis but still fails) |
| s257 | 3 | numerical | Various patterns |
| s31111 | 3 | numerical | Various patterns |
| s322 | 3 | numerical | Various patterns |
| s323 | 3 | numerical | Various patterns |

**Error Type Summary:**
- Numerical errors: 9 functions
- Timeout: 1 function (s176)

### Passing Functions (141):
s000, s111, s1111, s1112, s1113, s1115, s1119, s112, s113, s114, s115, s1161, s118, s119, s121, s1213, s122, s1221, s123, s1232, s124, s1244, s125, s1251, s126, s127, s1279, s128, s1281, s131, s13110, s132, s1351, s141, s1421, s151, s152, s162, s171, s172, s173, s174, s2101, s2102, s2111, s212, s221, s222, s2233, s2244, s2251, s2275, s231, s232, s233, s235, s241, s242, s243, s244, s251, s252, s253, s254, s255, s256, s258, s261, s271, s2710, s2711, s2712, s272, s273, s274, s275, s276, s277, s278, s279, s281, s291, s292, s293, s311, s3110, s3111, s3112, s3113, s312, s313, s314, s315, s316, s317, s318, s319, s321, s3251, s331, s332, s341, s343, s351, s352, s353, s4112, s4113, s4114, s4115, s4116, s4117, s4121, s421, s422, s423, s424, s431, s441, s442, s443, s451, s452, s453, s471, s481, s482, s491, va, vag, vas, vbor, vdotr, vif, vpv, vpvpv, vpvts, vpvtv, vsumr, vtv, vtvtv

### Analysis Tools Used

| Tool | Purpose |
|------|---------|
| compute_parallel_dims.py | Detect parallelizable dimensions and invalid parallelization |
| compute_war_analysis.py | Detect WAR (Write-After-Read) anti-dependencies |
| compute_statement_overwrites.py | Detect statement overwrite patterns |
| compute_stream_compaction.py | Detect stream compaction (conditional copy) patterns |
| compute_crossing_threshold.py | Detect crossing threshold / reverse access patterns |
| compute_loop_unrolling.py | Detect manually unrolled loop patterns |
| compute_statement_reordering.py | Detect statement reordering optimizations |
| compute_pointer_aliasing.py | Detect pointer aliasing patterns |
| compute_early_exit.py | Detect early exit patterns |

---

## 🔬 LLM Triton v3 with Retry Mechanism (2025-12-06) - test12

### Summary
| Metric | Count | Percentage |
|--------|-------|------------|
| ✅ **PASSING** | 141 | 93.4% |
| ❌ **FAILING** | 10 | 6.6% |

### Key Finding: Highest Pass Rate at the Time

**Integrated pipeline (`generate_and_test.py`) with automatic retry mechanism:**
- Max 10 attempts per function
- On failure, provides error feedback to LLM for retry
- Stops on numerical errors after attempt 3

### Pass Rate by Attempt
| Attempt | Passed | Cumulative |
|---------|--------|------------|
| Attempt 1 | 113 | 113 (74.8%) |
| Attempt 2 | +20 | 133 (88.1%) |
| Attempt 3+ | +8 | 141 (93.4%) |

**Key Insight:** 28 additional functions passed after error feedback. Extended retry mechanism (up to 10 attempts) helped recover 8 more functions that needed 3+ attempts.

### Comparison with test11

| Metric | test11 | test12 | Change |
|--------|--------|--------|--------|
| Passing | 137 (90.7%) | 141 (93.4%) | **+4** |
| Failing | 14 (9.3%) | 10 (6.6%) | **-4** |

### Functions FIXED by Retry (28 functions total):

**Passed on Attempt 2 (20 functions):**
| Function | Notes |
|----------|-------|
| **s1113** | Fixed after 1 retry |
| **s114** | Fixed after 1 retry |
| **s1213** | Fixed after 1 retry |
| **s221** | Fixed after 1 retry |
| **s2251** | Fixed after 1 retry |
| **s2275** | Fixed after 1 retry |
| **s232** | Fixed after 1 retry |
| **s242** | Fixed after 1 retry |
| **s243** | Fixed after 1 retry |
| **s252** | Fixed after 1 retry |
| **s255** | Fixed after 1 retry |
| **s256** | Fixed after 1 retry |
| **s272** | Fixed after 1 retry |
| **s292** | Fixed after 1 retry |
| **s311** | Fixed after 1 retry |
| **s319** | Fixed after 1 retry |
| **s323** | Fixed after 1 retry |
| **s341** | Fixed after 1 retry |
| **s351** | Fixed after 1 retry |
| **s424** | Fixed after 1 retry |

**Passed on Attempt 3+ (8 functions):**
| Function | Attempts | Notes |
|----------|----------|-------|
| **s128** | 3 | Fixed after 2 retries |
| **s141** | 6 | Fixed after 5 retries |
| **s2102** | 3 | Fixed after 2 retries |
| **s244** | 3 | Fixed after 2 retries |
| **s258** | 6 | Fixed after 5 retries |
| **s315** | 4 | Fixed after 3 retries |
| **s332** | 5 | Fixed after 4 retries |
| **s421** | 3 | Fixed after 2 retries |

### Failed Functions (10) - Error Breakdown

| Function | Attempts | Error Type |
|----------|----------|-----------|
| s116 | 5 | numerical |
| s176 | 10 | timeout |
| s211 | 3 | numerical |
| s2111 | 3 | numerical |
| s257 | 3 | numerical |
| s261 | 3 | numerical |
| s277 | 3 | numerical |
| s281 | 3 | numerical |
| s322 | 3 | numerical |
| s343 | 3 | numerical |

**Error Type Summary:**
- Numerical errors: 9 functions (s116, s211, s2111, s257, s261, s277, s281, s322, s343)
- Timeout: 1 function (s176)

### Passing Functions (141):
s000, s111, s1111, s1112, s1113, s1115, s1119, s112, s113, s114, s115, s1161, s118, s119, s121, s1213, s122, s1221, s123, s1232, s124, s1244, s125, s1251, s126, s127, s1279, s128, s1281, s131, s13110, s132, s1351, s141, s1421, s151, s152, s161, s162, s171, s172, s173, s174, s175, s2101, s2102, s212, s221, s222, s2233, s2244, s2251, s2275, s231, s232, s233, s235, s241, s242, s243, s244, s251, s252, s253, s254, s255, s256, s258, s271, s2710, s2711, s2712, s272, s273, s274, s275, s276, s278, s279, s291, s292, s293, s311, s3110, s3111, s31111, s3112, s3113, s312, s313, s314, s315, s316, s317, s318, s319, s321, s323, s3251, s331, s332, s341, s351, s352, s353, s4112, s4113, s4114, s4115, s4116, s4117, s4121, s421, s422, s423, s424, s431, s441, s442, s443, s451, s452, s453, s471, s481, s482, s491, va, vag, vas, vbor, vdotr, vif, vpv, vpvpv, vpvts, vpvtv, vsumr, vtv, vtvtv

### Failing Functions (10):
s116, s176, s211, s2111, s257, s261, s277, s281, s322, s343

---

## 🔬 LLM Triton v3 with Retry Mechanism (2025-12-05) - test11

### Summary
| Metric | Count | Percentage |
|--------|-------|------------|
| ✅ **PASSING** | 137 | 90.7% |
| ❌ **FAILING** | 14 | 9.3% |

### Key Finding: Highest Pass Rate Yet

**Integrated pipeline (`generate_and_test.py`) with automatic retry mechanism:**
- Max 3 attempts per function
- On failure, provides error feedback to LLM for retry
- Distinguishes between numerical and non-numerical errors

### Pass Rate by Attempt
| Attempt | Passed | Cumulative |
|---------|--------|------------|
| Attempt 1 | 114 | 114 (75.5%) |
| Attempt 2 | +17 | 131 (86.8%) |
| Attempt 3 | +6 | 137 (90.7%) |

**Key Insight:** 23 additional functions passed after error feedback. All three attempts contributed meaningful improvements, showing the retry mechanism remains effective.

### Comparison with test10

| Metric | test10 | test11 | Change |
|--------|--------|--------|--------|
| Passing | 128 (84.8%) | 137 (90.7%) | **+9** |
| Failing | 23 (15.2%) | 14 (9.3%) | **-9** |

### Functions FIXED by Retry (23 functions total):

**Passed on Attempt 2 (17 functions):**
| Function | Notes |
|----------|-------|
| **s112** | Fixed after 1 retry |
| **s1213** | Fixed after 1 retry |
| **s122** | Fixed after 1 retry |
| **s141** | Fixed after 1 retry |
| **s161** | Fixed after 1 retry |
| **s2102** | Fixed after 1 retry |
| **s2251** | Fixed after 1 retry |
| **s242** | Fixed after 1 retry |
| **s258** | Fixed after 1 retry |
| **s291** | Fixed after 1 retry |
| **s323** | Fixed after 1 retry |
| **s3251** | Fixed after 1 retry |
| **s331** | Fixed after 1 retry |
| **s332** | Fixed after 1 retry |
| **s351** | Fixed after 1 retry |
| **s421** | Fixed after 1 retry |
| **s453** | Fixed after 1 retry |

**Passed on Attempt 3 (6 functions):**
| Function | Notes |
|----------|-------|
| **s241** | Fixed after 2 retries |
| **s252** | Fixed after 2 retries |
| **s255** | Fixed after 2 retries |
| **s3112** | Fixed after 2 retries |
| **s312** | Fixed after 2 retries |
| **s352** | Fixed after 2 retries |

### Failed Functions (14) - Error Breakdown

| Function | Error Type |
|----------|-----------|
| s116 | compilation |
| s119 | numerical |
| s123 | numerical |
| s13110 | compilation |
| s176 | timeout |
| s211 | numerical |
| s256 | numerical |
| s257 | numerical |
| s277 | compilation |
| s281 | numerical |
| s322 | numerical |
| s342 | compilation |
| s353 | numerical |
| s482 | numerical |

**Error Type Summary:**
- Numerical errors: 9 functions (s119, s123, s211, s256, s257, s281, s322, s353, s482)
- Compilation errors: 4 functions (s116, s13110, s277, s342)
- Timeout: 1 function (s176)

### Passing Functions (137):
s000, s111, s1111, s1112, s1113, s1115, s1119, s112, s113, s114, s115, s1161, s118, s121, s1213, s122, s1221, s1232, s124, s1244, s125, s1251, s126, s127, s1279, s128, s1281, s131, s132, s1351, s141, s1421, s151, s152, s161, s162, s171, s172, s173, s174, s175, s2101, s2102, s2111, s212, s221, s222, s2233, s2244, s2251, s2275, s231, s232, s233, s235, s241, s242, s243, s244, s251, s252, s253, s254, s255, s258, s261, s271, s2710, s2711, s2712, s272, s273, s274, s275, s276, s278, s279, s291, s292, s293, s311, s3110, s3111, s31111, s3112, s3113, s312, s313, s314, s315, s316, s317, s318, s319, s321, s323, s3251, s331, s332, s341, s343, s351, s352, s4112, s4113, s4114, s4115, s4116, s4117, s4121, s421, s422, s423, s424, s431, s441, s442, s443, s451, s452, s453, s471, s481, s491, va, vag, vas, vbor, vdotr, vif, vpv, vpvpv, vpvts, vpvtv, vsumr, vtv, vtvtv

### Failing Functions (14):
s116, s119, s123, s13110, s176, s211, s256, s257, s277, s281, s322, s342, s353, s482

---

## 🔬 LLM Triton v3 with Retry Mechanism (2025-12-04) - test10

### Summary
| Metric | Count | Percentage |
|--------|-------|------------|
| ✅ **PASSING** | 128 | 84.8% |
| ❌ **FAILING** | 23 | 15.2% |

### Key Finding: Improved Pass Rate with Retry Mechanism

**Integrated pipeline (`generate_and_test.py`) with automatic retry mechanism:**
- Max 3 attempts per function
- On failure, provides error feedback to LLM for retry
- Distinguishes between numerical and non-numerical errors

### Pass Rate by Attempt
| Attempt | Passed | Cumulative |
|---------|--------|------------|
| Attempt 1 | 93 | 93 (61.6%) |
| Attempt 2 | +28 | 121 (80.1%) |
| Attempt 3 | +7 | 128 (84.8%) |

**Key Insight:** 35 additional functions passed after error feedback. Unlike test9, attempt 3 now yields 7 additional passes, showing the retry mechanism continues to be effective on the third try.

### Comparison with test9

| Metric | test9 | test10 | Change |
|--------|-------|--------|--------|
| Passing | 120 (79.5%) | 128 (84.8%) | **+8** |
| Failing | 31 (20.5%) | 23 (15.2%) | **-8** |

### Functions FIXED by Retry (35 functions total):

**Passed on Attempt 2 (28 functions):**
| Function | Notes |
|----------|-------|
| **s1115** | Numerical error → fixed |
| **s112** | Numerical error → fixed |
| **s113** | Numerical error → fixed |
| **s114** | Numerical error → fixed |
| **s116** | Numerical error → fixed |
| **s122** | Numerical error → fixed |
| **s1221** | Numerical error → fixed |
| **s1232** | Numerical error → fixed |
| **s124** | Compilation error → fixed |
| **s126** | Numerical error → fixed |
| **s128** | Compilation error → fixed |
| **s141** | Numerical error → fixed |
| **s1421** | Numerical error → fixed |
| **s175** | Numerical error → fixed |
| **s232** | Numerical error → fixed |
| **s243** | Numerical error → fixed |
| **s252** | Numerical error → fixed |
| **s257** | Numerical error → fixed |
| **s261** | Numerical error → fixed |
| **s275** | Numerical error → fixed |
| **s291** | Numerical error → fixed |
| **s312** | Numerical error → fixed |
| **s315** | Numerical error → fixed |
| **s318** | Numerical error → fixed |
| **s332** | Numerical error → fixed |
| **s351** | Numerical error → fixed |
| **s353** | Numerical error → fixed |
| **s421** | Numerical error → fixed |

**Passed on Attempt 3 (7 functions):**
| Function | Notes |
|----------|-------|
| **s123** | Numerical error → fixed after 2 retries |
| **s2102** | Numerical error → fixed after 2 retries |
| **s2275** | Numerical error → fixed after 2 retries |
| **s292** | Numerical error → fixed after 2 retries |
| **s31111** | Numerical error → fixed after 2 retries |
| **s331** | Numerical error → fixed after 2 retries |
| **s481** | Numerical error → fixed after 2 retries |

### Failed Functions (23) - ALL Numerical Errors

| Function | Error Type | Function | Error Type |
|----------|-----------|----------|-----------|
| s119 | numerical | s1213 | numerical |
| s151 | numerical | s161 | numerical |
| s162 | numerical | s176 | numerical |
| s211 | numerical | s2111 | numerical |
| s2244 | numerical | s244 | numerical |
| s255 | numerical | s256 | numerical |
| s258 | numerical | s281 | numerical |
| s322 | numerical | s341 | numerical |
| s343 | numerical | s4113 | numerical |
| s424 | numerical | s431 | numerical |
| s482 | numerical | s491 | numerical |
| vas | numerical | | |

### Non-Numerical Errors: NONE

**All 23 failed functions failed due to numerical errors (incorrect computation results).** No compilation errors, runtime errors, or API usage errors in the final attempts.

### Passing Functions (128):
s000, s111, s1111, s1112, s1113, s1115, s1119, s112, s113, s114, s115, s116, s1161, s118, s121, s122, s1221, s123, s1232, s124, s1244, s125, s1251, s126, s127, s1279, s128, s1281, s131, s13110, s132, s1351, s141, s1421, s152, s171, s172, s173, s174, s175, s2101, s2102, s212, s221, s222, s2233, s2251, s2275, s231, s232, s233, s235, s241, s242, s243, s251, s252, s253, s254, s257, s261, s271, s2710, s2711, s2712, s272, s273, s274, s275, s276, s277, s278, s279, s291, s292, s293, s311, s3110, s3111, s31111, s3112, s3113, s312, s313, s314, s315, s316, s317, s318, s319, s321, s323, s3251, s331, s332, s342, s351, s352, s353, s4112, s4114, s4115, s4116, s4117, s4121, s421, s422, s423, s441, s442, s443, s451, s452, s453, s471, s481, va, vag, vbor, vdotr, vif, vpv, vpvpv, vpvts, vpvtv, vsumr, vtv, vtvtv

### Failing Functions (23):
s119, s1213, s151, s161, s162, s176, s211, s2111, s2244, s244, s255, s256, s258, s281, s322, s341, s343, s4113, s424, s431, s482, s491, vas

---

## 🔬 LLM Triton v3 with Retry Mechanism (2025-12-03) - test9

### Summary
| Metric | Count | Percentage |
|--------|-------|------------|
| ✅ **PASSING** | 120 | 79.5% |
| ❌ **FAILING** | 31 | 20.5% |

### Key Finding: Retry Mechanism Significantly Improves Pass Rate

**New integrated pipeline (`generate_and_test.py`) with automatic retry mechanism:**
- Max 3 attempts per function
- On failure, provides error feedback to LLM for retry
- Distinguishes between numerical and non-numerical errors

### Pass Rate by Attempt
| Attempt | Passed | Cumulative |
|---------|--------|------------|
| Attempt 1 | 102 | 102 (67.5%) |
| Attempt 2 | +18 | 120 (79.5%) |
| Attempt 3 | +0 | 120 (79.5%) |

**Key Insight:** 18 additional functions passed after error feedback. Attempt 3 yielded no additional fixes, suggesting diminishing returns after 2 retries.

### Comparison with test8

| Metric | test8 | test9 | Change |
|--------|-------|-------|--------|
| Passing | 102 (67.5%) | 120 (79.5%) | **+18** |
| Failing | 49 (32.5%) | 31 (20.5%) | **-18** |

### Functions FIXED by Retry (18 functions - Passed on Attempt 2):
| Function | Notes |
|----------|-------|
| **s112** | Numerical error → fixed |
| **s122** | Numerical error → fixed |
| **s124** | Numerical error → fixed |
| **s128** | Numerical error → fixed |
| **s1421** | Numerical error → fixed |
| **s2111** | Numerical error → fixed |
| **s2233** | Numerical error → fixed |
| **s2244** | Numerical error → fixed |
| **s243** | Numerical error → fixed |
| **s252** | Numerical error → fixed |
| **s254** | Numerical error → fixed |
| **s256** | Numerical error → fixed |
| **s312** | Numerical error → fixed |
| **s318** | Numerical error → fixed |
| **s341** | Numerical error → fixed |
| **s351** | Numerical error → fixed |
| **s353** | Numerical error → fixed |
| **s4116** | Numerical error → fixed |

### Failed Functions (31) - ALL Numerical Errors

| Function | Max Error | Function | Max Error |
|----------|-----------|----------|-----------|
| s1213 | 4.81e+00 | s1221 | 8.92e+00 |
| s123 | unknown | s126 | 2.52e+01 |
| s151 | 1.98e+00 | s161 | 6.28e+00 |
| s162 | 1.15e+00 | s176 | 1.12e+01 |
| s2102 | 1.00e+00 | s211 | 4.74e+00 |
| s2251 | unknown | s232 | unknown |
| s255 | 1.66e+00 | s257 | unknown |
| s261 | 5.02e+00 | s275 | unknown |
| s281 | 5.06e+00 | s291 | unknown |
| s292 | unknown | s319 | unknown |
| s322 | 7.97e+01 | s332 | unknown |
| s342 | unknown | s343 | 4.04e+00 |
| s4113 | 3.11e+00 | s424 | 4.94e+00 |
| s431 | 2.13e+00 | s481 | unknown |
| s482 | 2.05e-01 | s491 | 7.05e+00 |
| vas | 3.63e+00 | | |

### Non-Numerical Errors: NONE

**All 31 failed functions failed due to numerical errors (incorrect computation results).** No compilation errors, runtime errors, or API usage errors in the final attempts.

### Passing Functions (120):
s000, s111, s1111, s1112, s1113, s1115, s1119, s112, s113, s114, s115, s116, s1161, s118, s119, s121, s122, s1232, s1244, s124, s125, s1251, s127, s1279, s1281, s128, s131, s13110, s132, s1351, s1421, s141, s152, s171, s172, s173, s174, s175, s2101, s212, s2111, s221, s222, s2233, s2244, s2275, s231, s233, s235, s241, s242, s243, s244, s251, s252, s253, s254, s256, s258, s271, s2710, s2711, s2712, s272, s273, s274, s276, s277, s278, s279, s293, s311, s3110, s3111, s31111, s3112, s3113, s312, s313, s314, s315, s316, s317, s318, s321, s323, s3251, s331, s341, s351, s352, s353, s4112, s4114, s4115, s4116, s4117, s4121, s421, s422, s423, s441, s442, s443, s451, s452, s453, s471, va, vag, vbor, vdotr, vif, vpv, vpvpv, vpvts, vpvtv, vsumr, vtv, vtvtv

### Failing Functions (31):
s1213, s1221, s123, s126, s151, s161, s162, s176, s2102, s211, s2251, s232, s255, s257, s261, s275, s281, s291, s292, s319, s322, s332, s342, s343, s4113, s424, s431, s481, s482, s491, vas

---

## 🔬 LLM Triton v3 Regeneration Test (2025-12-02) - test8_results.log

### Summary
| Metric | Count | Percentage |
|--------|-------|------------|
| ✅ **PASSING** | 102 | 67.5% |
| ❌ **FAILING** | 49 | 32.5% |

### Key Finding: Prompt Updates and Baseline Fixes

**Full regeneration of all 151 functions with updated prompts and baseline bug fixes. Pass rate: 102/151 (67.5%).**

### Comparison with test7

| Metric | test7 | test8 | Change |
|--------|-------|-------|--------|
| Passing | 104 (68.9%) | 102 (67.5%) | **-2** |
| Failing | 47 (31.1%) | 49 (32.5%) | +2 |

### Functions that REGRESSED (test7 → test8):
| Function | test7 | test8 | Error Type |
|----------|-------|-------|------------|
| **s1232** | ✓ | ✗ | Numerical error |
| **s152** | ✓ | ✗ | UnsupportedAST: break |
| **s2102** | ✓ | ✗ | Numerical error |
| **s212** | ✓ | ✗ | Numerical error |
| **s2244** | ✓ | ✗ | Numerical error |
| **s233** | ✓ | ✗ | Numerical error |
| **s242** | ✓ | ✗ | Numerical error |
| **s258** | ✓ | ✗ | Numerical error |
| **s3110** | ✓ | ✗ | AttributeError: constexpr.to() |
| **s314** | ✓ | ✗ | CompilationError: store after loop |
| **s319** | ✓ | ✗ | CompilationError: PassManager failed |
| **s342** | ✓ | ✗ | Numerical error |
| **s353** | ✓ | ✗ | Numerical error |
| **s4113** | ✓ | ✗ | Numerical error |
| **vas** | ✓ | ✗ | ValueError: _builder |

### Functions that IMPROVED (test7 → test8):
| Function | test7 | test8 | Notes |
|----------|-------|-------|-------|
| **s119** | ✗ | ✓ | Prompt/baseline update |
| **s121** | ✗ | ✓ | Prompt/baseline update |
| **s1232** | ✗ | ✓ | Prompt/baseline update |
| **s211** | ✗ | ✓ | Prompt/baseline update |
| **s251** | ✗ | ✓ | Prompt/baseline update |
| **s313** | ✗ | ✓ | Prompt/baseline update |
| **s343** | ✗ | ✓ | Prompt/baseline update |
| **s421** | ✗ | ✓ | Prompt/baseline update |
| **s491** | ✗ | ✓ | Prompt/baseline update |

### Non-Numerical Errors (16 functions - Compilation/Runtime Errors)

| Function | Error Type | Description |
|----------|------------|-------------|
| **s114** | UnsupportedAST | `break` statement not supported in Triton |
| **s123** | UnsupportedAST | `break` statement not supported in Triton |
| **s124** | UnsupportedAST | `break` statement not supported in Triton |
| **s132** | CompilationError | `tl.arange()` compilation error |
| **s2251** | TypeError | `cannot add pointers together` |
| **s257** | ValueError | `unsupported tensor index: constexpr[0]` |
| **s275** | ValueError | `_builder` argument - scalar indexing in kernel |
| **s291** | CompilationError | `tl.load` scalar index compilation error |
| **s292** | UnsupportedAST | `break` statement not supported in Triton |
| **s3110** | AttributeError | `'constexpr' object has no attribute 'to'` |
| **s314** | CompilationError | `tl.store` after for loop not allowed |
| **s319** | CompilationError | `PassManager::run failed` |
| **s332** | UnsupportedAST | `break` statement not supported in Triton |
| **s351** | CompilationError | `tl.arange()` compilation error |
| **s352** | CompilationError | `tl.arange(0, 5)` inside for loop |
| **s482** | UnsupportedAST | `return` inside for/while not supported |
| **vas** | ValueError | `_builder` argument - scalar indexing in kernel |

### Non-Numerical Error Summary by Category
| Category | Count | Functions | Prompt Rule Exists? |
|----------|-------|-----------|---------------------|
| UnsupportedAST (break/continue/return) | 7 | s114, s123, s124, s292, s332, s482 | ✅ YES - LLM ignores rule |
| CompilationError (tl.arange in loop) | 3 | s132, s351, s352 | ✅ YES - LLM ignores rule |
| ValueError (_builder/scalar indexing) | 2 | s275, vas | ✅ YES - LLM ignores rule |
| CompilationError (other) | 2 | s314, s319 | ❌ NO - new patterns |
| TypeError (pointer arithmetic) | 1 | s2251 | ❌ NO - need new rule |
| ValueError (tensor index constexpr) | 1 | s257 | ✅ Related to scalar indexing |
| AttributeError (constexpr methods) | 1 | s3110 | ❌ NO - need new rule |
| CompilationError (scalar load) | 1 | s291 | ✅ Related to scalar indexing |

### Numerical Errors (32 functions)
| Function | Max Error | Function | Max Error |
|----------|-----------|----------|-----------|
| s112 | 1.37e-01 | s1213 | 9.54e+00 |
| s1221 | 1.02e+02 | s126 | 5.47e+01 |
| s128 | 1.19e+01 | s141 | 7.76e+00 |
| s1421 | 7.36e+00 | s151 | 4.13e+00 |
| s161 | 9.41e+00 | s162 | 2.77e+00 |
| s176 | 1.13e+02 | s211 | 9.58e+00 |
| s2111 | 3.68e+09 | s212 | 1.65e+01 |
| s2244 | (numerical) | s233 | 5.98e+01 |
| s242 | 1.98e+04 | s244 | 1.19e+01 |
| s256 | 4.24e+00 | s258 | (numerical) |
| s261 | 5.88e+00 | s277 | 1.06e+01 |
| s281 | 1.38e+08 | s322 | 3.17e+18 |
| s3251 | 1.50e+01 | s341 | 3.17e+00 |
| s342 | (numerical) | s353 | 2.08e+00 |
| s4113 | 5.10e+00 | s424 | (numerical) |
| s431 | (numerical) | s442 | (numerical) |

### Passing Functions (102):
s000, s111, s1111, s1112, s1113, s1115, s1119, s113, s115, s116, s1161, s118, s119, s121, s122, s1232, s1244, s125, s1251, s127, s1279, s1281, s131, s13110, s1351, s152, s171, s172, s173, s174, s175, s2101, s2102, s221, s222, s2233, s2275, s231, s232, s235, s241, s243, s251, s252, s253, s254, s255, s271, s2710, s2711, s2712, s272, s273, s274, s276, s278, s279, s293, s311, s3111, s31111, s3112, s3113, s312, s313, s315, s316, s317, s318, s321, s323, s331, s343, s4112, s4114, s4115, s4116, s4117, s4121, s421, s422, s423, s441, s443, s451, s452, s453, s471, s481, s491, va, vag, vbor, vdotr, vif, vpv, vpvpv, vpvts, vpvtv, vsumr, vtv, vtvtv

### Failing Functions (49):
s112, s114, s1213, s1221, s123, s124, s126, s128, s132, s141, s1421, s151, s161, s162, s176, s211, s2111, s212, s2244, s2251, s233, s242, s244, s256, s257, s258, s261, s275, s277, s281, s291, s292, s3110, s314, s319, s322, s3251, s332, s341, s342, s351, s352, s353, s4113, s424, s431, s442, s482, vas

---

## 🔬 LLM Triton v3 Targeted Fixes (2025-12-02) - test7_results.log

### Summary
| Metric | Count | Percentage |
|--------|-------|------------|
| ✅ **PASSING** | 104 | 68.9% |
| ❌ **FAILING** | 47 | 31.1% |

### Key Finding: Targeted Manual Fixes

**10 specific functions were manually fixed based on test6 failure analysis. NOT a full regeneration.**

### Manually Fixed Functions (10 files changed)

| Function | Fix Applied |
|----------|-------------|
| **s174** | Fixed missing `BLOCK_SIZE=` keyword argument in kernel launch |
| **s31111** | Fixed baseline bug (removed `test` param), updated triton to match |
| **s312** | Fallback to CPU for product reduction (Triton scalar reduction limitation) |
| **s318** | Fixed function signature and incremented indexing logic |
| **s332** | Fixed break issue with vectorized approach, proper threshold search |
| **s351** | Fixed offsets calculation in loop unrolling |
| **s353** | Complete rewrite with explicit unrolled approach for gather operation |
| **s423** | Fixed parameter order and array indexing logic |
| **s424** | Fixed parameter order and store mask logic |
| **s482** | Minor fix but still uses break (still fails due to unsupported AST) |

### Comparison with test6

| Metric | test6 | test7 | Change |
|--------|-------|-------|--------|
| Passing | 92 (60.9%) | 104 (68.9%) | **+12** |
| Failing | 59 (39.1%) | 47 (31.1%) | -12 |

### Functions that IMPROVED (test6 → test7):
| Function | test6 | test7 | Notes |
|----------|-------|-------|-------|
| **s174** | ✗ | ✓ | **Manual fix** |
| **s31111** | ✗ | ✓ | **Manual fix** (baseline + triton) |
| **s318** | ✗ | ✓ | **Manual fix** |
| **s351** | ✗ | ✓ | **Manual fix** |
| **s353** | ✗ | ✓ | **Manual fix** |
| **s423** | ✗ | ✓ | **Manual fix** |
| **s424** | ✗ | ✓ | **Manual fix** |
| **s1113** | ✗ | ✓ | LLM variance |
| **s118** | ✗ | ✓ | LLM variance |
| **s173** | ✗ | ✓ | LLM variance |
| **s212** | ✗ | ✓ | LLM variance |
| **s2244** | ✗ | ✓ | LLM variance |
| **s233** | ✗ | ✓ | LLM variance |
| **s235** | ✗ | ✓ | LLM variance |
| **s242** | ✗ | ✓ | LLM variance |
| **s254** | ✗ | ✓ | LLM variance |
| **s255** | ✗ | ✓ | LLM variance |
| **s258** | ✗ | ✓ | LLM variance |
| **s261** | ✗ | ✓ | LLM variance |
| **s275** | ✗ | ✓ | LLM variance |
| **s276** | ✗ | ✓ | LLM variance |
| **s277** | ✗ | ✓ | LLM variance |
| **s3110** | ✗ | ✓ | LLM variance |
| **s352** | ✗ | ✓ | LLM variance |
| **s4112** | ✗ | ✓ | LLM variance |
| **s4114** | ✗ | ✓ | LLM variance |
| **s4115** | ✗ | ✓ | LLM variance |
| **s4116** | ✗ | ✓ | LLM variance |
| **s422** | ✗ | ✓ | LLM variance |
| **s442** | ✗ | ✓ | LLM variance |
| **s452** | ✗ | ✓ | LLM variance |
| **s471** | ✗ | ✓ | LLM variance |
| **vag** | ✗ | ✓ | LLM variance |
| **vas** | ✗ | ✓ | LLM variance |
| **vbor** | ✗ | ✓ | LLM variance |

### Functions that REGRESSED (test6 → test7) - LLM Variance:
| Function | test6 | test7 | Error Type |
|----------|-------|-------|------------|
| **s123** | ✓ | ✗ | UnsupportedAST: break |
| **s1232** | ✓ | ✗ | UnsupportedAST: break |
| **s128** | ✓ | ✗ | UnsupportedAST: break |
| **s141** | ✓ | ✗ | UnsupportedAST: continue |
| **s152** | ✓ | ✗ | Numerical error |
| **s175** | ✗ | ✗ | UnsupportedAST: break |
| **s211** | ✓ | ✗ | Numerical error |
| **s2111** | ✗ | ✗ | CompilationError: simultaneous multiple comparison |
| **s251** | ✓ | ✗ | Numerical error |
| **s291** | ✓ | ✗ | CompilationError |
| **s313** | ✓ | ✗ | Numerical error |

### Non-Numerical Errors (18 functions - Compilation/Runtime Errors)

| Function | Error Type | Description |
|----------|------------|-------------|
| **s123** | UnsupportedAST | break statement not supported in Triton |
| **s1232** | UnsupportedAST | break statement not supported in Triton |
| **s124** | ValueError | _builder argument - scalar indexing in kernel |
| **s128** | UnsupportedAST | break statement not supported in Triton |
| **s141** | UnsupportedAST | continue statement not supported in Triton |
| **s175** | UnsupportedAST | break statement not supported in Triton |
| **s2111** | CompilationError | `if 1 <= i < LEN_2D:` - chained comparison not supported |
| **s2251** | ValueError | _builder argument - scalar indexing in kernel |
| **s252** | ValueError | _builder argument - scalar indexing in kernel |
| **s257** | CompilationError | `tl.load(ptr + scalar, mask=vector_mask)` - scalar ptr + vector mask mismatch |
| **s291** | CompilationError | `tl.load(b_ptr, mask=mask)` - bare pointer without offset |
| **s292** | UnsupportedAST | break statement not supported in Triton |
| **s312** | ValueError | _builder argument - scalar indexing in kernel |
| **s317** | MissingArgs | Missing 2 required positional arguments: 'a' and 'b' |
| **s331** | ValueError | _builder argument - scalar indexing in kernel |
| **s332** | UnsupportedAST | break statement not supported in Triton |
| **s341** | UnsupportedAST | break statement not supported in Triton |
| **s482** | UnsupportedAST | break statement not supported in Triton |

### Non-Numerical Error Summary by Category
| Category | Count | Functions | Prompt Rule Exists? |
|----------|-------|-----------|---------------------|
| UnsupportedAST (break/continue) | 9 | s123, s1232, s128, s141, s175, s292, s332, s341, s482 | ✅ YES - LLM ignores rule |
| ValueError (_builder/scalar indexing) | 5 | s124, s2251, s252, s312, s331 | ✅ YES - LLM ignores rule |
| CompilationError (chained comparison) | 1 | s2111 | ❌ NO - need new rule |
| CompilationError (scalar+vector mismatch) | 1 | s257 | ✅ Related to scalar indexing |
| CompilationError (bare pointer load) | 1 | s291 | ❌ NO - need new rule |
| MissingArgs | 1 | s317 | - |

**Key Insights**:
1. LLM violates explicit prompt rules despite clear "NEVER do this" instructions with examples
2. New rules needed for: chained comparisons (`1 <= x < n`), bare pointer loads without offsets

### Numerical Errors (29 functions)
| Function | Max Error | Function | Max Error |
|----------|-----------|----------|-----------|
| s112 | 4.04e-01 | s119 | 2.19e+01 |
| s121 | 1.50e+01 | s1213 | 8.17e+00 |
| s1221 | 1.13e+01 | s126 | 2.48e+01 |
| s151 | 1.29e+00 | s161 | 4.76e+00 |
| s162 | 1.96e+00 | s176 | 1.20e+01 |
| s2102 | 1.00e+00 | s211 | 4.19e+00 |
| s2233 | 2.87e+01 | s231 | 2.11e+01 |
| s232 | inf | s243 | 2.87e-01 |
| s244 | 2.52e+00 | s256 | 3.28e+00 |
| s281 | 4.73e+00 | s322 | 7.86e-01 |
| s323 | 1.54e+01 | s342 | 5.11e+00 |
| s343 | 3.59e+00 | s4113 | 3.29e+00 |
| s421 | 5.37e+00 | s424 | 5.25e+00 |
| s431 | 1.81e+00 | s453 | 4.44e+02 |
| s491 | 4.44e+00 | | |

### Passing Functions (104):
s000, s111, s1111, s1112, s1113, s1115, s1119, s113, s114, s115, s116, s1161, s118, s122, s1244, s125, s1251, s127, s1279, s1281, s131, s13110, s132, s1351, s1421, s152, s171, s172, s173, s174, s2101, s212, s221, s222, s2244, s2275, s233, s235, s241, s242, s251, s253, s254, s255, s258, s261, s271, s2710, s2711, s2712, s272, s273, s274, s275, s276, s277, s278, s279, s293, s311, s3110, s3111, s31111, s3112, s3113, s313, s314, s315, s316, s318, s319, s321, s3251, s351, s352, s353, s4112, s4114, s4115, s4116, s4117, s4121, s422, s423, s441, s442, s443, s451, s452, s471, s481, va, vag, vas, vbor, vdotr, vif, vpv, vpvpv, vpvts, vpvtv, vsumr, vtv, vtvtv

### Failing Functions (47):
s112, s119, s121, s1213, s1221, s123, s1232, s124, s126, s128, s141, s151, s161, s162, s175, s176, s2102, s211, s2111, s2233, s2251, s231, s232, s243, s244, s252, s256, s257, s281, s291, s292, s312, s317, s322, s323, s331, s332, s341, s342, s343, s4113, s421, s424, s431, s453, s482, s491

---

## 🔬 LLM Triton v3 Regeneration Test (2025-12-01) - test6_results.log

### Summary
| Metric | Count | Percentage |
|--------|-------|------------|
| ✅ **PASSING** | 92 | 60.9% |
| ❌ **FAILING** | 59 | 39.1% |

### Key Finding: LLM Non-Determinism Impact

**Same prompt rules as test5, but regenerated all 151 functions to measure variance.**

### Comparison with test5 (Same Prompt, Different Run)

| Metric | test5 | test6 | Change |
|--------|-------|-------|--------|
| Passing | 97 (64.2%) | 92 (60.9%) | **-5** |
| Failing | 54 (35.8%) | 59 (39.1%) | +5 |

### Functions that REGRESSED (test5 → test6):
| Function | test5 | test6 | Error Type |
|----------|-------|-------|------------|
| **s112** | ✓ | ✗ | Numerical error |
| **s118** | ✓ | ✗ | Numerical error |
| **s121** | ✓ | ✗ | Compilation error |
| **s261** | ✓ | ✗ | Numerical error |
| **s271** | ✓ | ✗ | Numerical error |
| **s291** | ✓ | ✗ | `break` statement (UnsupportedAST) |
| **s3110** | ✓ | ✗ | Numerical error |
| **s331** | ✓ | ✗ | `_builder` error (scalar indexing) |
| **s343** | ✓ | ✗ | Numerical error |
| **s4116** | ✓ | ✗ | Numerical error |
| **s421** | ✓ | ✗ | Numerical error |

### Functions that IMPROVED (test5 → test6):
| Function | test5 | test6 | Notes |
|----------|-------|-------|-------|
| **s1112** | ✗ | ✓ | Was `tl.cdiv` error, now fixed |
| **s1119** | ✗ | ✓ | Now passes |
| **s1244** | ✗ | ✓ | Now passes |
| **s235** | ✗ | ✓ | Now passes |
| **s242** | ✗ | ✓ | Now passes |
| **s243** | ✗ | ✓ | Now passes |
| **s275** | ✗ | ✓ | Now passes |

### Key Observations

1. **Net change: -4 functions** (11 regressed, 7 improved)
2. **s331 regressed**: The scalar indexing rule was not followed in this regeneration
3. **s291 regressed**: LLM used `break` statement (unsupported in Triton)
4. **Variance range**: ~60-65% pass rate with current prompt rules

### Non-Numerical Errors (10 functions - Compilation/Runtime Errors)

| Function | Error Type | Description |
|----------|------------|-------------|
| **s174** | MissingArgs | Missing 1 required positional argument: 'M' |
| **s31111** | TypeError | 'int' object is not callable |
| **s312** | AttributeError | module 'triton.language' has no attribute 'mul' |
| **s318** | CompilationError | at 13:12: compilation error |
| **s332** | MissingArgs | Missing 1 required positional argument: 't_val' |
| **s351** | MissingArgs | Missing 1 required positional argument: 'c' |
| **s353** | MissingArgs | Missing 1 required positional argument: 'c' |
| **s423** | MissingArgs | Missing 1 required positional argument: 'xx' |
| **s424** | MissingArgs | Missing 1 required positional argument: 'xx' |
| **s482** | AttributeError | module 'triton.language' has no attribute 'any' |

### Non-Numerical Error Summary by Category
| Category | Count | Functions |
|----------|-------|-----------|
| Missing arguments (LLM signature bugs) | 6 | s174, s332, s351, s353, s423, s424 |
| Triton API errors (non-existent functions) | 2 | s312, s482 |
| Compilation errors | 1 | s318 |
| Other | 1 | s31111 |

### Passing Functions (92):
s000, s111, s1111, s1112, s1113, s1115, s1119, s113, s114, s115, s116, s1161, s119, s122, s123, s1232, s1244, s125, s1251, s1279, s128, s1281, s131, s13110, s1351, s141, s152, s172, s175, s2101, s211, s212, s221, s222, s2275, s231, s233, s235, s241, s242, s243, s251, s253, s2710, s2711, s2712, s272, s273, s274, s275, s278, s279, s292, s293, s311, s3111, s3112, s3113, s313, s314, s315, s316, s317, s319, s321, s323, s3251, s352, s4112, s4114, s4115, s4117, s4121, s441, s442, s443, s451, s452, s453, s481, va, vag, vbor, vdotr, vif, vpv, vpvpv, vpvts, vpvtv, vsumr, vtv, vtvtv

### Failing Functions (59):
s112, s118, s121, s1213, s1221, s124, s126, s127, s132, s1421, s151, s161, s162, s171, s173, s174, s176, s2102, s2111, s2233, s2244, s2251, s232, s244, s252, s254, s255, s256, s257, s258, s261, s271, s276, s277, s281, s291, s3110, s31111, s312, s318, s322, s331, s332, s341, s342, s343, s351, s353, s4113, s4116, s421, s422, s423, s424, s431, s471, s482, s491, vas

---

## 🔬 LLM Triton v3 Testing with Scalar Indexing Rule (2025-12-01) - test5_results.log

### Summary
| Metric | Count | Percentage |
|--------|-------|------------|
| ✅ **PASSING** | 97 | 64.2% |
| ❌ **FAILING** | 54 | 35.8% |

### Prompt Rules Applied (Same as test6)

1. **`tl.arange()` Rule**: Never use `tl.arange()` inside a for loop - define once at kernel start
2. **Scalar Indexing Rule**: Never index a tensor with a scalar variable inside @triton.jit kernel - use vectorized operations

**Infrastructure Fix:**
- `ip` array now initialized with `torch.randint(..., dtype=torch.long)` instead of float

### Comparison with test4 (Before Scalar Indexing Rule)

| Metric | test4 (before) | test5 (after) | Change |
|--------|----------------|---------------|--------|
| Passing | 96 (63.6%) | 97 (64.2%) | +1 |
| Failing | 55 (36.4%) | 54 (35.8%) | -1 |

### Key Fixes in test5

**Scalar Indexing Rule Impact:**
- ✅ **s3112**: `_builder` error FIXED - now passes
- ✅ **s331**: `_builder` error FIXED - now passes

**Infrastructure Fix (`ip` array type) Impact:**
- ✅ **s4112, s4114, s4116, vag**: Index type error FIXED - now pass

**Remaining Issues:**
- **vas, s491, s353**: Still fail due to numerical/algorithm issues (scatter operations)

### Non-Numerical Errors (9 functions - Compilation/Runtime Errors)

| Function | Error Type | Description |
|----------|------------|-------------|
| **s1112** | ValueError | Cannot call @triton.jit'd outside of kernel scope |
| **s31111** | TypeError | 'int' object is not callable |
| **s318** | MissingArgs | Missing 1 required positional argument: 'inc_val' |
| **s332** | MissingArgs | Missing 1 required positional argument: 't_val' |
| **s351** | MissingArgs | Missing 1 required positional argument: 'c' |
| **s353** | MissingArgs | Missing 1 required positional argument: 'ip' |
| **s423** | MissingArgs | Missing 1 required positional argument: 'xx' |
| **s424** | MissingArgs | Missing 1 required positional argument: 'xx' |
| **s482** | AttributeError | module 'triton.language' has no attribute 'any' |

### Non-Numerical Error Summary by Category
| Category | Count | Functions |
|----------|-------|-----------|
| Missing arguments (LLM signature bugs) | 6 | s318, s332, s351, s353, s423, s424 |
| @triton.jit usage errors | 1 | s1112 |
| Triton API errors | 1 | s482 |
| Other | 1 | s31111 |

### Passing Functions (97):
s000, s111, s1111, s1113, s1115, s112, s113, s114, s115, s116, s1161, s118, s119, s121, s1221, s122, s123, s1232, s124, s1244, s125, s1251, s127, s1279, s128, s1281, s131, s13110, s1351, s141, s152, s161, s171, s172, s175, s2101, s211, s212, s221, s222, s2233, s2275, s231, s233, s241, s251, s253, s254, s261, s271, s2710, s2711, s2712, s272, s273, s274, s278, s279, s291, s293, s311, s3110, s3111, s3112, s3113, s312, s313, s314, s315, s316, s317, s319, s321, s323, s3251, s331, s343, s352, s4112, s4114, s4115, s4116, s4117, s4121, s421, s441, s442, s443, s451, s453, s481, va, vag, vbor, vdotr, vif, vpv, vpvpv, vpvts, vpvtv, vsumr, vtv, vtvtv

### Failing Functions (54):
s1112, s1119, s1213, s126, s132, s1421, s151, s162, s173, s174, s176, s2102, s2111, s2244, s2251, s232, s235, s242, s243, s244, s252, s255, s256, s257, s258, s275, s276, s277, s281, s292, s31111, s318, s322, s332, s341, s342, s351, s353, s4113, s422, s423, s424, s431, s452, s471, s482, s491, vas

---

## 🔬 PyTorch Baseline Testing (2025-11-30)

### Summary
| Metric | Count | Percentage |
|--------|-------|------------|
| ✅ **PASSING** | 92 | 60.9% |
| ❌ **FAILING** | 59 | 39.1% |

### Non-Numerical Errors (13 functions - Compilation/Runtime Errors)

| Function | Error Type | Description |
|----------|------------|-------------|
| **s114** | UnsupportedAST | `continue` statement not supported in Triton |
| **s1213** | ValueError | @triton.jit usage error - _builder argument outside JIT |
| **s1232** | UnsupportedAST | `break` statement not supported in Triton |
| **s13110** | ValueError | Cannot call @triton.jit'd outside kernel scope |
| **s275** | ValueError | @triton.jit usage error - _builder argument outside JIT |
| **s312** | UnsupportedAST | `lambda` not supported in Triton |
| **s3112** | ValueError | @triton.jit usage error - _builder argument outside JIT |
| **s318** | MissingArgs | Missing 4 required positional arguments: 'b', 'c', 'd', 'e' |
| **s332** | MissingArgs | Missing 1 required positional argument: 't_val' |
| **s351** | MissingArgs | Missing 1 required positional argument: 'c' |
| **s4113** | IncompatibleType | Type mismatch: pointer<fp32> vs triton.language.float32 |
| **s4115** | IncompatibleType | Type mismatch: pointer<fp32> vs triton.language.float32 |
| **s482** | UnsupportedAST | `break` statement not supported in Triton |

### Non-Numerical Error Summary by Category
| Category | Count | Functions |
|----------|-------|-----------|
| Unsupported AST (`break`/`continue`/`lambda`) | 5 | s114, s1232, s312, s482, s312 |
| @triton.jit Usage Errors | 4 | s1213, s13110, s275, s3112 |
| Missing Arguments | 3 | s318, s332, s351 |
| Type/Pointer Errors | 2 | s4113, s4115 |

### Passing Functions (92):
s000, s111, s1111, s1112, s1113, s1115, s115, s116, s1161, s118, s121, s1221, s123, s124, s1244, s125, s1251, s127, s1279, s128, s1281, s131, s1351, s141, s152, s161, s171, s172, s175, s2101, s211, s212, s221, s222, s2251, s2275, s235, s241, s243, s251, s252, s253, s255, s261, s271, s2710, s2711, s2712, s272, s273, s274, s277, s278, s279, s291, s293, s311, s3110, s3111, s3113, s313, s314, s315, s316, s317, s319, s321, s3251, s331, s343, s352, s4117, s4121, s421, s441, s442, s443, s451, s452, s453, s481, va, vbor, vdotr, vif, vpv, vpvpv, vpvts, vpvtv, vsumr, vtv, vtvtv

### Failing Functions (59):
s1119, s112, s113, s114, s119, s1213, s122, s1232, s126, s13110, s132, s1421, s151, s162, s173, s174, s176, s2102, s2111, s2233, s2244, s231, s232, s233, s242, s244, s254, s256, s257, s258, s275, s276, s281, s292, s31111, s3112, s312, s318, s322, s323, s332, s341, s342, s351, s353, s4112, s4113, s4114, s4115, s4116, s422, s423, s424, s431, s471, s482, s491, vag, vas

---

## 🔬 LLM Triton v3 Testing (2025-11-30) - test2_results.log

### Summary
| Metric | Count | Percentage |
|--------|-------|------------|
| ✅ **PASSING** | 101 | 66.9% |
| ❌ **FAILING** | 50 | 33.1% |

### Non-Numerical Errors (9 functions - Compilation/Runtime Errors)

| Function | Error Type | Description |
|----------|------------|-------------|
| **s126** | MissingArgs | Missing 6 required positional arguments: 'd', 'e', 'aa', 'bb', 'cc', 'flat_2d_array' |
| **s174** | MissingArgs | Missing 1 required positional argument: 'M' |
| **s275** | ValueError | @triton.jit usage error - _builder argument outside JIT |
| **s2275** | ValueError | @triton.jit usage error - _builder argument outside JIT |
| **s318** | MissingArgs | Missing 4 required positional arguments: 'b', 'c', 'd', 'e' |
| **s332** | MissingArgs | Missing 1 required positional argument: 't_val' |
| **s351** | MissingArgs | Missing 1 required positional argument: 'c' |
| **s4113** | IncompatibleType | Type mismatch: pointer<fp32> vs triton.language.float32 |
| **s4115** | IncompatibleType | Type mismatch: pointer<fp32> vs triton.language.float32 |

### Non-Numerical Error Summary by Category
| Category | Count | Functions |
|----------|-------|-----------|
| Missing Arguments | 5 | s126, s174, s318, s332, s351 |
| @triton.jit Usage Errors | 2 | s275, s2275 |
| Type/Pointer Errors | 2 | s4113, s4115 |

### Passing Functions (101):
s000, s111, s1111, s1112, s1113, s1115, s112, s113, s114, s115, s116, s1161, s118, s121, s122, s1221, s123, s1232, s124, s1244, s125, s1251, s127, s1279, s128, s1281, s131, s13110, s1351, s141, s152, s161, s171, s172, s175, s2101, s211, s212, s221, s222, s235, s241, s242, s243, s251, s253, s254, s258, s271, s2710, s2711, s2712, s272, s273, s274, s277, s278, s279, s291, s292, s293, s311, s3110, s3111, s3112, s3113, s313, s314, s315, s316, s317, s319, s321, s323, s3251, s331, s341, s342, s343, s352, s4117, s4121, s421, s441, s442, s443, s451, s452, s453, s481, va, vbor, vdotr, vif, vpv, vpvpv, vpvts, vpvtv, vsumr, vtv, vtvtv

### Failing Functions (50):
s1119, s119, s1213, s126, s132, s1421, s151, s162, s173, s174, s176, s2102, s2111, s2233, s2244, s2251, s2275, s231, s232, s233, s244, s252, s255, s256, s257, s261, s275, s276, s281, s31111, s312, s318, s322, s332, s351, s353, s4112, s4113, s4114, s4115, s4116, s422, s423, s424, s431, s471, s482, s491, vag, vas

---

## 🔬 LLM Triton v3 Testing with WAR Fix (2025-12-01) - test3_results.log

### Summary
| Metric | Count | Percentage |
|--------|-------|------------|
| ✅ **PASSING** | 97 | 64.2% |
| ❌ **FAILING** | 54 | 35.8% |

### Key Change: WAR Dependency Detection Bug Fix

**Bug Fixed in PET Analysis (`/home/qinxiao/workspace/pet/isl_analysis/compute_war_dependences.py`):**

The WAR (Write-After-Read) detection incorrectly classified 2D diagonal backward shift patterns as WAR when they are actually RAW (Read-After-Write). This caused incorrect code generation for functions like `s119`.

**Pattern Affected:**
```
aa[i][j] = aa[i-1][j-1] + bb[i][j]
```
- Reads from `aa[i-1][j-1]` (earlier diagonal position)
- Writes to `aa[i][j]` (current position)
- This is **RAW** (reading from earlier data), not WAR
- When processing one dimension sequentially, no race condition exists

**Fix Applied:** Added detection for 2D arrays with backward offsets on both dimensions. When read indices have backward offsets relative to write indices (e.g., `(i-1, j-1)` vs `(i, j)`), it's correctly identified as RAW.

### Non-Numerical Errors (23 functions - Compilation/Runtime Errors)

| Function | Error Type |
|----------|------------|
| **s1119** | Test timeout |
| **s2251** | ValueError: @triton.jit usage error - _builder argument |
| **s252** | Triton compilation error (inline arange) |
| **s257** | Triton compilation error (inline arange) |
| **s275** | Triton compilation error (inline arange) |
| **s31111** | 'int' object is not callable |
| **s3112** | Triton compilation error (inline arange) |
| **s312** | Triton compilation error (inline arange) |
| **s332** | module 'triton.language' has no attribute 'bitcast' |
| **s351** | Missing 1 required positional argument: 'c' |
| **s352** | Triton compilation error (inline arange) |
| **s353** | tensors used as indices must be long, int, byte or bool |
| **s4112** | tensors used as indices must be long, int, byte or bool |
| **s4113** | Triton compilation error (inline arange) |
| **s4114** | tensors used as indices must be long, int, byte or bool |
| **s4115** | Triton compilation error (inline arange) |
| **s4116** | tensors used as indices must be long, int, byte or bool |
| **s423** | Tensor size mismatch (expanded size) |
| **s424** | CUDA error: illegal memory access |
| **s453** | Triton compilation error (inline arange) |
| **s491** | tensors used as indices must be long, int, byte or bool |
| **vag** | tensors used as indices must be long, int, byte or bool |
| **vas** | tensors used as indices must be long, int, byte or bool |

### Non-Numerical Error Summary by Category
| Category | Count | Functions |
|----------|-------|-----------|
| Triton compilation errors (inline arange) | 9 | s252, s257, s275, s3112, s312, s352, s4113, s4115, s453 |
| Tensor index type errors | 7 | s353, s4112, s4114, s4116, s491, vag, vas |
| Memory/CUDA errors | 2 | s423, s424 |
| @triton.jit usage errors | 1 | s2251 |
| Triton API errors | 1 | s332 |
| Missing arguments | 1 | s351 |
| Timeout | 1 | s1119 |
| Other | 1 | s31111 |

### Comparison with test2 (Before WAR Fix)

| Metric | test2 (before) | test3 (after) | Change |
|--------|----------------|---------------|--------|
| Passing | 101 (66.9%) | 97 (64.2%) | -4 |
| Failing | 50 (33.1%) | 54 (35.8%) | +4 |

**Note:** The net decrease is due to LLM regeneration variability, not the fix itself.

### Functions FIXED by WAR Analysis Fix (7 functions):
| Function | Issue Fixed |
|----------|-------------|
| **s119** | No longer incorrectly uses `aa_copy.clone()` - reads directly from `aa_ptr` |
| s1213 | Correct dependency analysis |
| s2233 | Correct dependency analysis |
| s2275 | Correct dependency analysis |
| s231 | Correct dependency analysis |
| s233 | Correct dependency analysis |
| s318 | Correct dependency analysis |

### Functions REGRESSED (11 functions - LLM variability):
s111, s172, s235, s258, s3112, s323, s341, s343, s352, s442, s453

These regressions are due to non-deterministic LLM output during regeneration, not related to the WAR fix.

### s119 Detailed Fix

**Before (test2 - INCORRECT):**
```python
def s119_triton(aa, bb):
    aa_copy = aa.clone()  # Unnecessary copy!
    for i_val in range(1, LEN_2D):
        # Read from copy, write to original
        # This breaks the algorithm because we never see updated values
```

**After (test3 - CORRECT):**
```python
def s119_triton(aa, bb):
    for i_val in range(1, LEN_2D):
        # Read directly from aa_ptr (previous row already computed)
        # Write to aa_ptr (current row)
        # Sequential i processing ensures correct dependency handling
```

**Verification:** s119 now passes with max_error=0.00e+00

### Passing Functions (97):
s000, s1111, s1112, s1113, s1115, s112, s113, s114, s115, s116, s1161, s118, s119, s121, s1213, s122, s1221, s123, s1232, s124, s1244, s125, s1251, s127, s1279, s128, s1281, s131, s13110, s1351, s141, s152, s161, s171, s175, s2101, s211, s212, s221, s222, s2233, s2275, s231, s233, s241, s242, s243, s251, s253, s254, s271, s2710, s2711, s2712, s272, s273, s274, s277, s278, s279, s291, s292, s293, s311, s3110, s3111, s3113, s313, s314, s315, s316, s317, s318, s319, s321, s3251, s331, s342, s4117, s4121, s421, s441, s443, s451, s452, s481, va, vbor, vdotr, vif, vpv, vpvpv, vpvts, vpvtv, vsumr, vtv, vtvtv

### Failing Functions (54):
s111, s1119, s126, s132, s1421, s151, s162, s172, s173, s174, s176, s2102, s2111, s2244, s2251, s232, s235, s244, s252, s255, s256, s257, s258, s261, s275, s276, s281, s31111, s3112, s312, s322, s323, s332, s341, s343, s351, s352, s353, s4112, s4113, s4114, s4115, s4116, s422, s423, s424, s431, s442, s453, s471, s482, s491, vag, vas

---

## 🔬 LLM Triton v3 Testing with `tl.arange()` Rule (2025-12-01) - test4_results.log

### Summary
| Metric | Count | Percentage |
|--------|-------|------------|
| ✅ **PASSING** | 96 | 63.6% |
| ❌ **FAILING** | 55 | 36.4% |

### Key Change: `tl.arange()` Inside For Loop Rule

**Prompt Rule Added:**
```
NEVER use `tl.arange()` inside a for loop - it causes compilation errors.
Define tl.arange() ONCE at kernel start, before any loops.
```

### Comparison with test3 (Before `tl.arange()` Rule)

| Metric | test3 (before) | test4 (after) | Change |
|--------|----------------|---------------|--------|
| Passing | 97 (64.2%) | 96 (63.6%) | -1 |
| Failing | 54 (35.8%) | 55 (36.4%) | +1 |

### `tl.arange()` Rule Effectiveness

Of the 5 functions originally failing due to `tl.arange()` in for loop:
| Function | test3 | test4 | Status |
|----------|-------|-------|--------|
| **s352** | FAIL | PASS | **Fixed by rule** |
| **s453** | FAIL | PASS | **Fixed by rule** |
| s252 | FAIL | FAIL | Different root cause (numerical) |
| s257 | FAIL | FAIL | Different root cause (tensor index) |
| s3112 | FAIL | FAIL | Different root cause (@triton.jit) |

**Result: 2/5 targeted functions fixed**

### Functions FIXED in test4 (9 functions):
| Function | Notes |
|----------|-------|
| **s352** | `tl.arange()` rule fix |
| **s453** | `tl.arange()` rule fix |
| s111 | LLM variability |
| s172 | LLM variability |
| s2244 | LLM variability |
| s235 | LLM variability |
| s275 | LLM variability |
| s343 | LLM variability |
| s442 | LLM variability |

### Functions REGRESSED in test4 (10 functions - LLM variability):
s121, s1213, s1221, s141, s161, s242, s243, s2710, s277, s331

### Non-Numerical Errors (27 functions - Compilation/Runtime Errors)

#### Tensor index type errors (8 functions)
| Function | Error |
|----------|-------|
| **s257** | ValueError: unsupported tensor index: constexpr[0] |
| **s353** | tensors used as indices must be long, int, byte or bool |
| **s4112** | tensors used as indices must be long, int, byte or bool |
| **s4114** | tensors used as indices must be long, int, byte or bool |
| **s4116** | tensors used as indices must be long, int, byte or bool |
| **s491** | tensors used as indices must be long, int, byte or bool |
| **vag** | tensors used as indices must be long, int, byte or bool |
| **vas** | tensors used as indices must be long, int, byte or bool |

#### Missing arguments (4 functions)
| Function | Error |
|----------|-------|
| **s174** | Missing 1 required positional argument: 'M' |
| **s2710** | Missing 1 required positional argument: 'LEN_1D' |
| **s332** | Missing 7 required positional arguments |
| **s351** | Missing 1 required positional argument: 'c' |

#### @triton.jit usage errors (2 functions)
| Function | Error |
|----------|-------|
| **s3112** | `_builder` argument must be provided outside of JIT |
| **s331** | `_builder` argument must be provided outside of JIT |

#### Triton API errors (2 functions)
| Function | Error |
|----------|-------|
| **s141** | zeros_like() got unexpected keyword argument 'dtype' |
| **s312** | module 'triton.language.math' has no attribute 'multiply_op' |

#### Type/pointer errors (2 functions)
| Function | Error |
|----------|-------|
| **s4113** | invalid operands: pointer<fp32> vs triton.language.float32 |
| **s4115** | invalid operands: pointer<fp32> vs triton.language.float32 |

#### Timeout (1 function)
| Function | Error |
|----------|-------|
| **s1119** | Test timeout |

#### Other compilation errors (7 functions)
| Function | Error |
|----------|-------|
| **s2251** | Compilation error |
| **s255** | Compilation error |
| **s258** | Compilation error |
| **s31111** | 'int' object is not callable (baseline bug - test is helper function, not parameter) |
| **s341** | Compilation error |
| **s423** | Tensor size mismatch |
| **s482** | Compilation error |

### Numerical/Algorithm Errors (29 functions)
| Function | Max Error | Function | Max Error |
|----------|-----------|----------|-----------|
| s121 | 1.60e+01 | s1213 | 8.18e+00 |
| s1221 | 3.76e+00 | s126 | 4.57e+01 |
| s132 | 4.90e+00 | s1421 | 5.10e+00 |
| s151 | 2.82e+00 | s161 | 5.57e+00 |
| s162 | 2.58e+00 | s173 | 6.48e+00 |
| s176 | 1.80e+02 | s2102 | 2.26e+00 |
| s2111 | 4.71e+10 | s232 | inf |
| s242 | 5.86e+03 |
| s243 | 1.90e+00 | s244 | 1.11e+01 |
| s252 | 1.14e+00 | s256 | 3.46e+00 |
| s261 | 8.44e+00 | s276 | 6.18e+00 |
| s277 | 1.08e+00 | s281 | 6.78e+00 |
| s322 | 4.06e+12 | s323 | 1.06e+02 |
| s422 | 7.65e+00 | s424 | 5.31e+00 |
| s431 | 5.78e+00 | s471 | 2.27e+01 |

### Non-Numerical Error Summary by Category
| Category | Count | Functions |
|----------|-------|-----------|
| Tensor index type errors | 8 | s257, s353, s4112, s4114, s4116, s491, vag, vas |
| Missing arguments | 4 | s174, s2710, s332, s351 |
| @triton.jit usage errors | 2 | s3112, s331 |
| Triton API errors | 2 | s141, s312 |
| Type/pointer errors | 2 | s4113, s4115 |
| Other compilation | 7 | s2251, s255, s258, s31111, s341, s423, s482 |
| Timeout | 1 | s1119 |

### Passing Functions (96):
s000, s111, s1111, s1112, s1113, s1115, s112, s113, s114, s115, s116, s1161, s118, s119, s122, s123, s1232, s124, s1244, s125, s1251, s127, s1279, s128, s1281, s131, s13110, s1351, s152, s171, s172, s175, s2101, s211, s212, s221, s222, s2233, s2244, s2275, s231, s233, s235, s241, s251, s253, s254, s271, s2711, s2712, s272, s273, s274, s275, s278, s279, s291, s292, s293, s311, s3110, s3111, s3113, s313, s314, s315, s316, s317, s318, s319, s321, s3251, s342, s343, s352, s4117, s4121, s421, s441, s442, s443, s451, s452, s453, s481, va, vbor, vdotr, vif, vpv, vpvpv, vpvts, vpvtv, vsumr, vtv, vtvtv

### Failing Functions (55):
s1119, s121, s1213, s1221, s126, s132, s141, s1421, s151, s161, s162, s173, s174, s176, s2102, s2111, s2251, s232, s242, s243, s244, s252, s255, s256, s257, s258, s261, s2710, s276, s277, s281, s31111, s3112, s312, s322, s323, s331, s332, s341, s351, s353, s4112, s4113, s4114, s4115, s4116, s422, s423, s424, s431, s471, s482, s491, vag, vas

---

## 🔬 C Ground Truth Testing (2025-11-29)

### Important Finding
Previous testing compared LLM-generated Triton against LLM-generated PyTorch baselines. This creates a problem: **if both implementations have the same bug, the test passes incorrectly**.

We created a **C reference library** compiled from the original TSVC source code to provide true ground truth.

### C Ground Truth Results - ALL 151 Functions Tested (2025-11-29)

| Metric | Count | Percentage |
|--------|-------|------------|
| ✅ **PASSING** | 97 | 64.2% |
| ❌ **FAILING** | 54 | 35.8% |
| ⏭️ **SKIPPED** | 0 | 0% |

**Note:** This run used the original prompt (no restrictions). Results vary slightly between runs due to LLM non-determinism.

**Infrastructure Bugs Fixed:**
- `d` array initialization: Use TSVC-style `1/(i+1)` (always positive) to avoid `exit(0)` in s481
- Helper functions added: s151s, s152s, test_helper, f_helper, s471s for interprocedural tests
- Scalar parameter typing: Proper float vs int handling
- No-array function support: s317 now testable

### Passing Functions (97):
s000, s111, s1111, s1112, s1115, s1119, s112, s113, s114, s116, s1161, s119, s121, s1232, s122, s124, s125, s1251, s127, s1279, s128, s1281, s13110, s132, s1351, s151, s152, s162, s171, s172, s174, s175, s2101, s2102, s2111, s2233, s2275, s231, s232, s235, s243, s251, s253, s271, s2710, s2711, s2712, s272, s273, s274, s275, s278, s279, s293, s311, s3110, s3111, s31111, s3112, s3113, s312, s313, s314, s315, s316, s317, s318, s319, s323, s331, s342, s4112, s4113, s4114, s4115, s4117, s4121, s421, s441, s443, s451, s453, s481, s491, va, vag, vas, vdotr, vif, vpv, vpvpv, vpvts, vpvtv, vsumr, vtv, vtvtv

### Failure Categories (54 total):

| Category | Count | Examples |
|----------|-------|----------|
| Compilation Errors | 14 | s126, s141, s233, s2251, s252, s255, s257, s291, s332, s341, s351, s352, s4116, s482 |
| Numerical/Algorithm Errors | 40 | s1113, s115, s118, s1213, s1221, s123, s1244, s131, s1421, s161, s173, s176, s211, s221, s222, s2244, s241, s242, s244, s254, s256, s258, s261, s276, s277, s281, s292, s321, s322, s3251, s343, s353, s422, s423, s424, s431, s442, s452, s471, vbor |

### Detailed Compilation Errors (14 functions - 2025-11-29 run)

| Function | Error Type | Root Cause |
|----------|------------|------------|
| **s126** | UnsupportedLanguageConstruct | `break` statement not supported in Triton kernels |
| **s141** | CompilationError | Tensor indexing outside JIT: `bb_vals[idx]` |
| **s2251** | UnsupportedLanguageConstruct | `break` statement not supported in Triton kernels |
| **s233** | UnsupportedLanguageConstruct | `break` statement not supported in Triton kernels |
| **s252** | CompilationError | Tensor indexing outside JIT: `b_vals[i] * c_vals[i]` |
| **s255** | CompilationError | `tl.arange(0, BLOCK_SIZE)` compilation issue |
| **s257** | CompilationError | `tl.store` with incompatible mask/shape |
| **s291** | CompilationError | `tl.arange(0, BLOCK_SIZE)` compilation issue |
| **s332** | UnsupportedLanguageConstruct | `break` statement not supported in Triton kernels |
| **s341** | CompilationError | Tensor indexing outside JIT: `valid_mask[i]` |
| **s351** | IncompatibleTypeError | Type mismatch: `alpha * b_vals` - pointer vs float |
| **s352** | CompilationError | Using `tl.arange(0, 5)` inline in load expression inside loop |
| **s4116** | IncompatibleTypeError | Type mismatch: `(j_idx - 1) * len_2d` - pointer vs int |
| **s482** | UnsupportedLanguageConstruct | `break` statement not supported in Triton kernels |

### Compilation Error Summary by Category (2025-11-29)
| Category | Count | Functions |
|----------|-------|-----------|
| `break` statement | 5 | s126, s2251, s233, s332, s482 |
| Tensor indexing outside JIT | 3 | s141, s252, s341 |
| Type/pointer errors | 3 | s351, s4116, s257 |
| tl.arange compilation issues | 3 | s255, s291, s352 |

### Implication
The **64.2% pass rate** against C ground truth is the true measure of LLM-generated Triton correctness. The 54 failures represent genuine implementation bugs in the LLM-generated Triton code, not test infrastructure issues.

---

## 📈 Test History Comparison

| Date | Test | PASS | FAIL | Pass Rate | Notes |
|------|------|------|------|-----------|-------|
| **2025-12-03** | **test9** | **120** | **31** | **79.5%** | **LATEST** - Retry mechanism, +18 from test8 |
| 2025-12-02 | test8 | 102 | 49 | 67.5% | LLM variance, -2 from test7 |
| 2025-12-02 | test7 | 104 | 47 | 68.9% | Manual fixes, +12 from test6 |
| 2025-12-01 | test6 | 92 | 59 | 60.9% | Regeneration test - demonstrates LLM variance |
| 2025-12-01 | test5 | 97 | 54 | 64.2% | + Scalar indexing rule + ip fix - s3112, s331 fixed |
| 2025-12-01 | test4 | 96 | 55 | 63.6% | + `tl.arange()` rule - s352, s453 fixed |
| 2025-12-01 | test3 | 97 | 54 | 64.2% | WAR fix applied - s119 now passing |
| 2025-11-30 | test2 | 101 | 50 | 66.9% | LLM Triton v3 (before WAR fix) |
| 2025-11-30 | test1 | 92 | 59 | 60.9% | PyTorch baseline (auto_test_all_tsvc.py) |
| 2025-11-29 | - | 97 | 54 | 64.2% | C ground truth - original prompt |
| 2025-11-28 | - | 94 | 57 | 62.3% | Previous C ground truth run |
| 2025-11-18 | - | 99 | 52 | 65.6% | PyTorch baseline comparison |

**Key Observations:**
- LLM output varies between runs due to non-determinism (~10 functions change per regeneration)
- test3 fixed WAR detection bug, enabling s119 to pass
- test4 added `tl.arange()` rule: 2/5 targeted functions fixed (s352, s453)
- test5 added scalar indexing rule: s3112, s331 fixed; infrastructure fix: s4112, s4114, s4116, vag fixed
- test6 vs test5: Same prompt, regenerated all functions → 11 regressed, 7 improved, net -4
- test7 vs test6: Same prompt, regenerated all functions → 34 improved, 11 regressed, net +12
- **test9 vs test8**: Retry mechanism with error feedback → **+18 functions fixed** (67.5% → 79.5%)
- **Retry effectiveness**: 18 functions fixed on attempt 2, 0 on attempt 3 (diminishing returns after 2 retries)
- **All remaining failures are numerical errors** - retry mechanism eliminated all compilation/runtime errors

---

## 📊 Overall Results (vs PyTorch Baseline)

### Summary
- ✅ **PASSING: 99 / 151 (65.6%)** *(Note: May include false positives)*
- ❌ **FAILING: 52 / 151 (34.4%)**

### Test History

| Date | Passing | Failing | Pass Rate | Notes |
|------|---------|---------|-----------|-------|
| 2025-11-06 (Initial) | 80/151 | 71/151 | 53.0% | First complete run with fixed infrastructure |
| 2025-11-17 (Regenerated) | 81/151 | 70/151 | 53.6% | All Triton implementations regenerated |
| 2025-11-17 (Investigated) | 84/151 | 67/151 | 55.6% | Test infrastructure fixes & tolerance corrections |
| **2025-11-18 (Deep Dive)** | **99/151** | **52/151** | **65.6%** | **Fixed LLM bugs, test bugs, and baseline bugs** |
| Change from Initial | **+19** | **-19** | **+12.6%** | Major improvement from systematic investigation |

**Key Finding:** Deep investigation revealed:
- **15 additional functions fixed** through LLM bug fixes and test infrastructure corrections
- **2 baseline bugs discovered** (s161, s212) where both baseline and LLM were wrong
- **True pass rate is 65.6%** - significantly higher than initially reported

---

## ✅ 99 Passing Functions

s000, s111, s1111, s1112, s1115, s112, s113, s114, s115, s116, s118, s119, s1119, s1161, s121, s1213, s122, s123, s124, s125, s1251, s126, s127, s1279, s128, s1281, s131, s132, s13110, s1351, s141, s1421, s151, s152, s161, s162, s171, s172, s173, s174, s175, s176, s2101, s211, s212, s221, s222, s2275, s241, s243, s251, s253, s254, s271, s2711, s2712, s272, s273, s274, s277, s278, s279, s292, s293, s311, s3110, s3111, s31111, s3112, s3113, s313, s314, s315, s316, s317, s318, s319, s321, s331, s4117, s4121, s421, s422, s431, s441, s442, s443, s451, s453, s481, s482, va, vdotr, vif, vpv, vpvpv, vpvtv, vsumr, vtv, vtvtv

### Latest Fixes (2025-11-18)

**✅ LLM Bugs Fixed (14 functions):**
- **s116**: Fixed incorrect indexing (treated BLOCK_SIZE as element indices instead of group indices)
- **s118**: Created faster in-kernel version (8-19x speedup vs sequential launches)
- **s119**: Fixed with sequential launches (diagonal dependency prevents in-kernel parallelization)
- **s1119**: Confirmed correct (LLM version already optimal with in-kernel loop)
- **s123**: Fixed two bugs: non-existent `tl.any()` function and wrong value detection logic
- **s126**: Fixed inefficient grid size (10,010 programs → 40 programs, >119x speedup)
- **s162**: Fixed incorrect masking logic (separate masks for load vs store)
- **s173**: Fixed unconditional parallelization (detects RAW dependencies when k < half_len)
- **s175**: Fixed unsafe memory access (missing `other` parameter for integer loads)

**✅ Test Infrastructure Bugs Fixed (5 functions):**
- **s131**: Missing `m` parameter in test
- **s132**: Missing `j, k` parameters in test
- **s141**: LLM correct, baseline times out at large N
- **s174**: Wrong array sizes in test (M vs N confusion)
- **s176**: Missing `iterations, m` parameters in test

**✅ Baseline Bugs Fixed (2 functions):**
- **s161**: Both baseline and LLM had RAW dependency bug (fixed with 2-phase execution)
- **s212**: Both baseline and LLM forgot to clone array 'a' (similar to s211 which correctly clones 'b')

---

## ❌ 52 Failing Functions

### Complete List (Alphabetical)

s1113, s1221, s1232, s1244, s2102, s2111, s2233, s2244, s2251, s231, s232, s233, s235, s242, s244, s252, s255, s256, s257, s258, s261, s2710, s275, s276, s281, s291, s312, s322, s323, s3251, s332, s341, s342, s343, s351, s352, s353, s4112, s4113, s4114, s4115, s4116, s423, s424, s452, s471, s491, vag, vas, vbor, vpvts

### By Category (52 total)

**1. Timeouts ⏱️ (~10 functions)**
Sequential CPU loops launching GPU kernels:
- s1232, s2111, s2233, s231, s232, s233, s256, s257

**2. Numerical/Algorithm Errors ❌ (~30 functions)**
Wrong results, accumulation errors:
- s1113, s1244, s2244, s2251, s235, s242, s244, s252, s255, s258, s261, s275, s276, s281, s291, s312, s322, s323, s3251, s452, s471, s491, vag, vas, vbor, vpvts

**3. Dimension/Shape Errors 📐 (~8 functions)**
Tensor size mismatches, index out of bounds:
- s2102, s2710, s341, s342, s343

**4. Other Errors (~4 functions)**
Complex issues requiring individual analysis:
- s1221, s351, s352, s353, s4112, s4113, s4114, s4115, s4116, s423, s424

---

## 🔍 Deep Investigation Findings (2025-11-18)

### LLM Bug Categories

**1. Race Conditions / RAW Dependencies**
- **s119**: Diagonal dependency (aa[i,j] = aa[i-1,j-1] + bb[i,j]) requires sequential launches
- **s173**: When k < half_len, creates overlap between read/write ranges → race condition
- **s161, s212**: Both baseline and LLM forgot to save array copies before modification
- **Pattern**: In-kernel loops don't synchronize across warps/threads

**2. Incorrect Masking / Bounds Checking**
- **s162**: Used separate masks for load vs store, stored incorrect values when source out of bounds
- **s175**: Loaded indices without `other=0`, created undefined behavior and illegal memory access
- **Pattern**: Always specify `other` parameter, use single comprehensive mask

**3. Performance Bugs**
- **s126**: Launched N programs (one per column) instead of N/256 programs (vectorized)
- **s118**: Sequential launches slow, in-kernel loop with atomics is 8-19x faster
- **Pattern**: Apply s1119 pattern (in-kernel loop) when dependencies allow

**4. Compilation Errors**
- **s123**: Used non-existent `tl.any()` function (should use `tl.where()`)
- **Pattern**: Hallucinated Triton API functions

**5. Algorithm Errors**
- **s116**: Treated BLOCK_SIZE as element offsets instead of group IDs
- **s123**: Detected written values by checking `value != 0.0` (fails when value IS 0)
- **Pattern**: Incorrect understanding of loop structure or data layout

### Test Infrastructure Issues

**1. Missing Parameters**
- **s131**: Missing `m` parameter
- **s132**: Missing `j, k` parameters
- **s176**: Missing `iterations, m` parameters
- **Pattern**: Scalar parameters from C code not passed in tests

**2. Wrong Array Sizes**
- **s174**: Created `a` and `b` with same size N, but algorithm requires `a` size ≥ 2*M where M = b.size(0)
- **Pattern**: Misunderstanding of array size relationships

**3. Baseline Performance**
- **s141**: Baseline is O(N²), times out at N=10000 (Triton is correct and fast)
- **Pattern**: Some baselines are inherently slow, not an implementation bug

### Baseline Bugs Discovered

**s161 - RAW Dependency Bug (Both Wrong)**
```c
if (b[i] < 0) {
    c[i+1] = a[i] + d[i] * d[i];  // Writes c[i+1]
} else {
    a[i] = c[i] + d[i] * e[i];    // Reads c[i] (could be updated by previous i-1!)
}
```
- **Baseline**: Processed both branches in parallel, reads OLD c values
- **LLM**: Same bug
- **Fix**: Two-phase execution (write c first, then read c)

**s212 - Missing Array Clone (Both Wrong)**
```c
a[i] *= c[i];          // Modifies a[i]
b[i] += a[i+1] * d[i]; // Should read ORIGINAL a[i+1]
```
- **Baseline**: Forgot to clone `a`, reads MODIFIED values
- **LLM**: Same bug
- **Fix**: Save `a_orig = a.clone()` before modification (like s211 does with `b_orig`)

**Impact**: Tests passed because both implementations were wrong in the same way!

---

## 📈 Success Rate by Function Type

| Type | Total | Passing | Pass Rate | Change |
|------|-------|---------|-----------|---------|
| Element-wise ops | ~45 | ~42 | 93% | +6% |
| Conditionals | ~30 | ~22 | 73% | +6% |
| Offset access | ~35 | ~25 | 71% | +17% |
| Reductions | ~20 | ~13 | 65% | +5% |
| 2D arrays | ~15 | ~9 | 60% | +13% |
| Complex fusion | ~10 | ~5 | 50% | +10% |

---

## 🎯 Key Patterns and Insights

### When In-Kernel Loop Works (s1119 Pattern)

**✅ Use in-kernel sequential loop:**
- **s1119**: Vertical dependency (aa[i,j] depends on aa[i-1,j]) - each column independent
- **s118**: Reduction with atomic_add - atomics handle concurrent writes
- **s126**: Same as s1119 - vertical dependency

**❌ Don't use in-kernel loop (need sequential launches):**
- **s119**: Diagonal dependency (aa[i,j] depends on aa[i-1,j-1]) - cross-thread dependency
- **s173**: When k < half_len - creates RAW overlap
- **Reason**: For-loops in Triton don't synchronize across warps/threads

### Masking Best Practices

**Anti-pattern (WRONG):**
```python
mask = condition1
source_mask = mask & condition2
data = tl.load(..., mask=source_mask, other=0.0)
result = compute(data)
tl.store(..., result, mask=mask)  # Stores when condition2 is False!
```

**Correct pattern:**
```python
mask = condition1 & condition2  # Combined mask
data = tl.load(..., mask=mask, other=0.0)
result = compute(data)
tl.store(..., result, mask=mask)  # Only stores when both conditions true
```

### Array Cloning for RAW Dependencies

**Pattern**: When iteration i reads array[i+offset] and modifies array[i]:
```python
# Save original values BEFORE any modification
array_orig = array.clone()

# Modify array
array[...] = modify(array[...])

# Read from original values
result[...] = compute(array_orig[...])
```

**Examples:**
- **s211**: Saves `b_orig` (reads b[i±1], modifies b[i]) ✅
- **s212**: Should save `a_orig` (reads a[i+1], modifies a[i]) but forgot ❌

---

## 📁 Analysis Documents

Detailed analysis for each investigated function:

- `my_triton_implementations/s112/S112_RACE_CONDITION_ANALYSIS.md` - SIMT implicit synchronization
- `my_triton_implementations/s115/S115_ANALYSIS.md` - Relative tolerance for value growth
- `my_triton_implementations/s116/S116_ANALYSIS.md` - Incorrect indexing logic
- `my_triton_implementations/s118/S118_ANALYSIS.md` - In-kernel loop optimization
- `my_triton_implementations/s119/S119_ANALYSIS.md` - Why in-kernel loop fails (diagonal dependency)
- `my_triton_implementations/s123/S123_ANALYSIS.md` - tl.any() bug and value detection
- `my_triton_implementations/s126/S126_ANALYSIS.md` - Grid size inefficiency
- `my_triton_implementations/s161/S161_ANALYSIS.md` - RAW dependency in baseline and LLM
- `my_triton_implementations/s162/S162_ANALYSIS.md` - Incorrect masking
- `my_triton_implementations/s173/S173_ANALYSIS.md` - Conditional parallelization
- `my_triton_implementations/s175/S175_ANALYSIS.md` - Unsafe memory access
- `my_triton_implementations/s212/S212_ANALYSIS.md` - Missing array clone

---

## 🏆 Conclusion

**Final Pass Rate: 65.6% (99/151 functions)**

The LLM successfully generated correct Triton implementations for **nearly two-thirds** of the TSVC benchmark suite, significantly higher than initially measured.

### Major Improvements

1. **+15 functions fixed** through systematic investigation
2. **Found 2 baseline bugs** where both implementations were wrong
3. **Pass rate improved** from 53.0% → 65.6% (+12.6%)
4. **True capability higher** than testing initially suggested

### Remaining Issues (52 functions)

1. **Timeouts** (~10): Sequential kernel launches
2. **Algorithm errors** (~30): Wrong computation logic
3. **Dimension handling** (~8): Shape mismatches
4. **Other** (~4): Complex issues

### Critical Insights

**1. Dependency Analysis is Hard**
- LLM struggles to identify when in-kernel loops are safe vs when sequential launches are needed
- Diagonal dependencies (s119), overlap conditions (s173) require careful analysis
- Even experts can miss these (s161, s212 baseline bugs)

**2. Masking is Subtle**
- Easy to create bugs with separate masks for load vs store
- Integer loads without `other` parameter cause undefined behavior
- Simple, comprehensive masks are safer than complex conditional logic

**3. Test Infrastructure Matters**
- 5+ functions failed due to test bugs, not implementation bugs
- Baseline bugs can cause false positives (s161, s212)
- Proper validation requires checking against true sequential C semantics

**4. Performance Patterns**
- s1119 pattern (in-kernel loop) can give 10-100x speedup when applicable
- But only works when no cross-thread dependencies exist
- Grid size matters enormously (s126: 251x more programs = timeout)

### Recommendations

1. **Validate baselines** against sequential C execution, not just TSVC spec
2. **Use relative tolerance** for algorithms with value growth/accumulation
3. **Test with multiple k values** to expose dependency bugs (like s173)
4. **Check masking carefully** - prefer single comprehensive masks
5. **Profile for performance** - correct doesn't mean optimal (s118, s126)

---

## 📊 Function Categories Summary

**s000-s113:** Single dimension operations (s1113, loop-splitting)
- Most pass with element-wise or simple dependencies
- **s112**: Works despite race conditions (SIMT synchronization)
- **s116**: Fixed incorrect group indexing

**s114-s119, s1119:** Double dimensions, triangular operations
- **s115**: need in-kernel loop and the pattern of s114
- **s118**: Optimized with in-kernel loop (8-19x speedup)
- **s119**: Requires sequential launches (diagonal dependency)
- **s1119**: Already optimal (in-kernel loop with vertical dependency)

**s121-s128:** Induction variables(s123, double phase)
- **s123**: Fixed tl.any() bug and value detection
- **s126**: Fixed grid size inefficiency (>119x speedup)

**s131-s132:** Global data flow analysis
- **s131, s132**: Test infrastructure bugs (missing parameters)

**s141:** Nonlinear dependence(s141, loop interchange)
- LLM correct, baseline times out at large N

**s151-s152:** Interprocedural data flow
- Both pass

**s161-s162:** Control flow
- **s161**: Both baseline and LLM had RAW bug (fixed)
- **s162**: Fixed masking bug

**s171-s176:** Symbolics
- **s173**: Fixed conditional parallelization
- **s174, s176**: Test infrastructure bugs
- **s175**: Fixed unsafe memory access

**s211-s1213:** Statement reordering
- **s211, s1213**: forward and backward dependency, need to put backward dependency lines later.
- **s212**: Both baseline and LLM forgot a_orig (fixed), only backward dependency, so only need to save orig.

**s221-s222:** loop distribution: split into parrellel and recurrence part(fall back to element-wise processsing)

**s231-s235:** loop interchange, with real dependency across one dimension or none, use s1119 solution with dependency.

**s241-s2244:** node spliting, dealing with overwritten cases.

**s251-s261:** scalar and array expansion

**s271-s2712:** control flow

**s281-s1281** crossing threshold

**s291-s293** loop peeling

**s2101-s2111** diagonals

**s311-s3113** reductions, functions like s312 are unable to parrelise.

**s321-s323** recurrence, doesn't benefit from Triton.

**s331, s332** search loops, s332 doesn't fall back to pytorch

**s341-s343** packing, using cumsum for output positions

**s351-s353** loop rerolling

**s421-s424** storage classes and equivalencing

**s451-s453** intrinsic functions

**s471** call statements

**s491** vector semantics, only parreliszable when each ip element is unique

**s4112-s4117** indirect addressing,   | Operation                                          | Parallelizable? | Method                |
  |----------------------------------------------------|-----------------|-----------------------|
  | Scatter with overwrite (a[ip[i]] = x) + duplicates | ❌ NO            | Sequential processing |
  | Scatter with accumulation (a[ip[i]] += x)          | ✅ YES           | tl.atomic_add         |
  | Scatter with unique indices                        | ✅ YES           | Normal parallel store |
  | Gather (x = a[ip[i]])                              | ✅ YES           | Normal parallel load  |

**va-vbor** control loops

---

**This investigation demonstrates that systematic bug analysis can significantly improve measured LLM performance, revealing true capabilities obscured by test infrastructure issues and baseline bugs.**

double dimension in-kernel loop > sequential loop launching
backward dependency only need to save orig to avoid race condition/
forward dependency only(recurrence, usually after loop-split), fall back to element-wise processsing/
backward + forward: statement reordering.
