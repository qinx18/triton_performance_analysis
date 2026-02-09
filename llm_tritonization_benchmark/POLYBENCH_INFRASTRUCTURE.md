# Polybench/C Triton Generation Pipeline — Infrastructure Overview

## Purpose

Extend the TSVC-based LLM Tritonization pipeline (151 kernels, 99.3% pass rate) to **Polybench/C 4.2.1** (30 kernels) — a more challenging benchmark with parametric bounds, deeper nesting, scalar parameters, and multi-dimensional arrays.

---

## Architecture

```
                    ┌─────────────────────────┐
                    │   Polybench/C 4.2.1     │
                    │  (30 HPC kernels)       │
                    └──────────┬──────────────┘
                               │
                    extract_polybench_kernels.py
                               │
                    ┌──────────▼──────────────┐
                    │  kernels_polybench/      │
                    │  30 standalone .c files  │
                    │  (#pragma scop format)   │
                    └──────────┬──────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
     ┌────────▼───────┐ ┌─────▼──────┐ ┌───────▼────────┐
     │ PET + ISL      │ │ LLVM 17    │ │ C Reference    │
     │ 16 analysis    │ │ Fallback   │ │ 30 .so libs    │
     │ modules        │ │ Adapters   │ │ via ctypes     │
     └────────┬───────┘ └─────┬──────┘ └───────┬────────┘
              └────────┬──────┘                 │
                       │                        │
              ┌────────▼──────────┐             │
              │ polybench_        │             │
              │ functions_db.py   │             │
              │ (30 kernel specs) │             │
              └────────┬──────────┘             │
                       │                        │
              ┌────────▼──────────────────┐     │
              │ generate_and_test_        │     │
              │ polybench.py              │     │
              │                           │     │
              │  1. Build analysis prompt │     │
              │  2. LLM generates Triton  │     │
              │  3. Correctness test ◄────┼─────┘
              │  4. Retry on failure      │
              │  (max 5 attempts)         │
              └───────────────────────────┘
```

---

## File Map

### Kernel Extraction & Analysis (`/home/qinxiao/workspace/pet/isl_analysis/`)

| File | Role |
|------|------|
| `extract_polybench_kernels.py` | Extracts 30 kernels from Polybench source → standalone `.c` files with `#pragma scop` |
| `kernels_polybench/` | 30 extracted kernel `.c` files (SMALL_DATASET sizes) |
| `llvm_analyzer.py` | Unified LLVM 17.0.0 analysis: AST, IR, DependenceAnalysis, SCEV |
| `llvm_fallback_adapters.py` | Drop-in LLVM replacements for PET modules (same return format) |
| `compute_war_dependences.py` | WAR (Write-After-Read) dependency detection |
| `compute_statement_overwrites.py` | Statement overwrite detection |
| `compute_stream_compaction.py` | Stream compaction pattern detection |
| `compute_parallel_dims.py` | Parallelizable dimension analysis |
| `compute_scalar_expansion.py` | Scalar expansion candidate detection |
| `compute_reduction_type.py` | Reduction pattern classification |
| `compute_pointer_aliasing.py` | Pointer aliasing analysis |
| `compute_loop_interchange.py` | Loop interchange opportunity detection |
| `compute_indirect_addressing.py` | Indirect addressing pattern detection |
| `compute_goto_conversion.py` | Goto-to-structured conversion |
| `compute_early_exit.py` | Early exit pattern detection |
| `compute_loop_unrolling.py` | Loop unrolling analysis |
| `compute_crossing_threshold.py` | Crossing threshold detection |
| `compute_convolution_pattern.py` | Convolution pattern detection |
| `compute_loop_distribution.py` | Loop distribution analysis |
| `compute_statement_reordering.py` | Statement reordering analysis |

### Pipeline (`llm_tritonization_benchmark/`)

| File | Role |
|------|------|
| `generate_and_test_polybench.py` | Main Polybench pipeline: prompt → LLM → test → retry |
| `generate_and_test.py` | Original TSVC pipeline (151 kernels) |
| `utilities/polybench_functions_db.py` | 30 kernel specs: arrays, shapes, scalar params, loop code |
| `utilities/tsvc_functions_db.py` | 151 TSVC kernel specs |
| `c_reference/polybench_reference.py` | Compile & load Polybench `.so` libraries |
| `c_reference/polybench_libs/` | 30 precompiled shared libraries (`libgemm.so`, etc.) |

---

## 30 Polybench/C Kernels

| Category | Kernels |
|----------|---------|
| **Datamining** | correlation, covariance |
| **Linear Algebra / BLAS** | gemm, gemver, gesummv, symm, syr2k, syrk, trmm, 2mm, 3mm |
| **Linear Algebra / Solvers** | cholesky, durbin, gramschmidt, lu, ludcmp, trisolv |
| **Linear Algebra / Kernels** | atax, bicg, doitgen, mvt |
| **Medley** | deriche, floyd_warshall, nussinov |
| **Stencils** | adi, fdtd_2d, heat_3d, jacobi_1d, jacobi_2d, seidel_2d |

---

## Analysis Pipeline Robustness

After fixing kernel extraction bugs and adding LLVM fallbacks:

| Module | Pass Rate | Notes |
|--------|-----------|-------|
| WAR | 100% (30/30) | PET + LLVM fallback |
| Overwrites | 100% (30/30) | PET + LLVM fallback |
| Stream | 100% (30/30) | PET + LLVM fallback |
| Aliasing | 100% (30/30) | PET-based |
| ParDims | 100% (30/30) | PET + LLVM fallback |
| Reduction | 100% (30/30) | PET-based |
| IndirAddr | 100% (30/30) | PET-based |
| Goto | 100% (30/30) | PET-based |
| ScalarExp | 90% (27/30) | 3 kernels empty (gemver, mvt, seidel_2d) |
| Interchange | 3% (1/30) | TSVC-specific, expected |
| Crossing, Unrolling, EarlyExit, Reordering, Convolution, LoopDist | 0% | TSVC-specific patterns, not in Polybench |

**0 crashes across 480 test combinations (16 modules x 30 kernels).**

---

## Key Technical Decisions

### C Reference via ctypes

Static global arrays in `.so` files require `CArrayType.in_dll()`:

```python
# Correct — direct reference to static global array
CType = ctypes.c_float * (NI * NJ)
c_arr = CType.in_dll(lib, 'C')
ctypes.memmove(c_arr, src.ctypes.data, src.nbytes)       # write
result = np.frombuffer(c_arr, dtype=np.float32).copy()    # read
```

### Digit-Starting Kernel Names (`2mm`, `3mm`)

- C function: `k2mm_kernel` (prefix `k`)
- Python import: `importlib.import_module("....2mm.attempt1")`
- Triton function: `k2mm_triton`

### Variable Renaming (`deriche`)

`y1` conflicts with POSIX Bessel function in `math.h`. Renamed to `yy1` via `name_remap` in extraction.

### Scalar Parameters

Only true **inputs** (alpha, beta, float_n, eps) are listed in `scalar_params`. Computed temporaries (a1-a8, nrm, w, sum, temp2) are excluded — they're computed inside the kernel.

---

## LLVM Fallback Strategy

When PET analysis fails (complex multi-statement scops), LLVM provides equivalent analysis:

| Capability | LLVM Tool | Replaces |
|-----------|-----------|----------|
| C AST parsing | `clang -ast-dump=json` | Regex-based parsing |
| Dependency analysis | `opt -passes='print<da>'` | PET WAR/RAW detection |
| Loop analysis | `opt -passes='print<scalar-evolution>'` | PET loop bounds |
| Array access patterns | LLVM IR analysis | PET access relations |

The `try_with_llvm_fallback()` function tries PET first, falls back to LLVM on failure.

---

## Running the Pipeline

```bash
# Process all 30 kernels
python generate_and_test_polybench.py

# Process specific kernels
python generate_and_test_polybench.py gemm lu atax

# Compile C reference libraries (if not already built)
python c_reference/polybench_reference.py
```

**Requirements**: `ANTHROPIC_API_KEY` environment variable, PyTorch + Triton (GPU), LLVM 17.0.0 (`/usr/local/bin/clang`, `/usr/local/bin/opt`).

---

## Comparison with TSVC Pipeline

| | TSVC | Polybench |
|---|---|---|
| Kernels | 151 | 30 |
| Complexity | Simple loops, fixed arrays | Parametric bounds, multi-dim arrays, scalar params |
| Array naming | Fixed (`a,b,c,d,e,aa,bb`) | Descriptive (`A,B,C,data,corr,mean`) |
| Sizes | `LEN_1D`=32000, `LEN_2D`=256 | Per-kernel (20-250) |
| LLM model | claude-sonnet-4-20250514 | claude-sonnet-4-20250514 |
| Max attempts | 3 | 5 |
| Pass rate | 99.3% (150/151) | 60.0% (18/30) |
| First-try pass | — | 10/30 (33.3%) |
| After retry | — | 8 additional |
| Script | `generate_and_test.py` | `generate_and_test_polybench.py` |

---

## First Run Results (2026-02-09)

**Model**: claude-sonnet-4-20250514 | **Max attempts**: 5

### Summary

| Metric | Count | Rate |
|--------|-------|------|
| Triton generated | 30/30 | 100% |
| Tests passed | 18/30 | 60.0% |
| Passed first try | 10/30 | 33.3% |
| Passed after retry | 8/30 | 26.7% |
| Failed (exhausted) | 12/30 | 40.0% |

### Per-Kernel Results

| Kernel | Attempts | Result | Failure Mode |
|--------|----------|--------|-------------|
| 2mm | 1 | PASS | |
| 3mm | 2 | PASS | |
| adi | 2 | PASS | |
| atax | 5 | FAIL | Compilation — repeated invalid Triton code |
| bicg | 1 | PASS | |
| cholesky | 1 | PASS | |
| correlation | 4 | PASS | |
| covariance | 5 | FAIL | Numerical (max_error 0.3–3.7) |
| deriche | 3 | PASS | |
| doitgen | 5 | FAIL | Numerical (3D array indexing) |
| durbin | 5 | FAIL | Numerical (max_error 2e-3, nearly passed) |
| fdtd_2d | 2 | PASS | |
| floyd_warshall | 5 | FAIL | Numerical (integer min operation) |
| gemm | 1 | PASS | |
| gemver | 5 | FAIL | Numerical (max_error 5e-4 on last try, nearly passed) |
| gesummv | 1 | PASS | |
| gramschmidt | 5 | FAIL | Numerical (sequential norm, 0.5–8.9) |
| heat_3d | 1 | PASS | |
| jacobi_1d | 5 | FAIL | Numerical (stencil WAR, 5e-7 to 0.05) |
| jacobi_2d | 5 | FAIL | Numerical (stencil WAR, 1.1–1.6) |
| lu | 1 | PASS | |
| ludcmp | 5 | FAIL | Numerical (sequential solver, 5–680) |
| mvt | 3 | PASS | |
| nussinov | 4 | PASS | |
| seidel_2d | 5 | FAIL | Numerical (Gauss-Seidel ordering, 0.07–0.1) |
| symm | 1 | PASS | |
| syr2k | 2 | PASS | |
| syrk | 1 | PASS | |
| trisolv | 5 | FAIL | Numerical (forward substitution, 0 to 1e37) |
| trmm | 1 | PASS | |

### Failure Analysis

The 12 failing kernels fall into clear categories:

**Sequential dependency chains** (6 kernels): durbin, gramschmidt, ludcmp, trisolv, jacobi_1d, seidel_2d — these have loop-carried dependencies where each iteration depends on the previous one's result, making GPU parallelization fundamentally difficult.

**Stencil WAR hazards** (2 kernels): jacobi_2d, seidel_2d — in-place stencils where reads and writes overlap, requiring double-buffering the LLM didn't implement correctly.

**Complex data flow** (3 kernels): covariance, doitgen, floyd_warshall — multi-phase kernels with intricate data dependencies across phases.

**Compilation failure** (1 kernel): atax — repeated Triton compilation errors (likely prompt issue).

**Near misses**: gemver (5e-4), durbin (2e-3), jacobi_1d (5e-7) — these nearly passed the 1e-3 threshold.
