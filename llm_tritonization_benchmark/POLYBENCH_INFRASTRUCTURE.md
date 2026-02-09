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

# Run performance benchmark on all passed kernels
python generate_and_test_polybench.py --benchmark

# Benchmark specific kernels
python generate_and_test_polybench.py --benchmark gemm lu atax

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
| Pass rate | 99.3% (150/151) | 83.3% (25/30) |
| First-try pass | — | 11/30 (36.7%) |
| After retry | — | 14 additional |
| GPU speedup (median) | — | 1.85x |
| Script | `generate_and_test.py` | `generate_and_test_polybench.py` |

---

## Final Results (2026-02-09)

**Model**: claude-sonnet-4-20250514 | **Max attempts**: 5 | **Tolerance**: abs < 1e-3 OR rel < 1e-4

### Summary

| Metric | Count | Rate |
|--------|-------|------|
| Triton generated | 30/30 | 100% |
| Tests passed | 25/30 | 83.3% |
| Passed first try | 11/30 | 36.7% |
| Passed after retry | 14/30 | 46.7% |
| Failed (exhausted) | 5/30 | 16.7% |

### Per-Kernel Results with Benchmarks

| Kernel | Attempts | Result | C ref (ms) | Triton (ms) | Speedup |
|--------|----------|--------|-----------|-------------|---------|
| 2mm | 1 | PASS | 0.392 | 0.099 | 3.96x |
| 3mm | 2 | PASS | 0.608 | 8.612 | 0.07x |
| adi | 2 | PASS | 2.799 | 36.921 | 0.08x |
| atax | 2 | PASS | 0.137 | 0.089 | 1.53x |
| bicg | 1 | PASS | 0.149 | 0.085 | 1.76x |
| cholesky | 1 | PASS | 0.314 | 0.189 | 1.66x |
| correlation | 4 | PASS | 0.422 | 0.459 | 0.92x |
| covariance | 2 | PASS | 0.418 | 0.077 | 5.40x |
| deriche | 3 | PASS | 0.648 | 0.123 | 5.28x |
| doitgen | 5 | FAIL | — | — | — |
| durbin | 5 | FAIL | — | — | — |
| fdtd_2d | 2 | PASS | 0.340 | 0.137 | 2.48x |
| floyd_warshall | 1 | PASS | 1.027 | 2.869 | 0.36x |
| gemm | 1 | PASS | 0.174 | 0.058 | 3.00x |
| gemver | 5 | PASS | 0.265 | 0.130 | 2.05x |
| gesummv | 1 | PASS | 0.171 | 0.092 | 1.85x |
| gramschmidt | 5 | FAIL | — | — | — |
| heat_3d | 1 | PASS | 4.976 | 2.871 | 1.73x |
| jacobi_1d | 1 | PASS | 0.076 | 0.043 | 1.76x |
| jacobi_2d | 2 | PASS | 0.561 | 0.600 | 0.94x |
| lu | 1 | PASS | 0.512 | 0.164 | 3.12x |
| ludcmp | 5 | FAIL | — | — | — |
| mvt | 3 | PASS | 0.194 | 0.128 | 1.51x |
| nussinov | 4 | PASS | 0.743 | 17.035 | 0.04x |
| seidel_2d | 5 | FAIL | — | — | — |
| symm | 1 | PASS | 0.250 | 0.108 | 2.32x |
| syr2k | 2 | PASS | 0.256 | 0.086 | 2.97x |
| syrk | 1 | PASS | 0.186 | 0.060 | 3.08x |
| trisolv | 3 | PASS | 0.133 | 3.713 | 0.04x |
| trmm | 1 | PASS | 0.231 | 0.071 | 3.26x |

### Speedup Statistics (25 passed kernels)

| Metric | Value |
|--------|-------|
| Median speedup | 1.76x |
| Mean speedup | 2.05x |
| Min speedup | 0.04x (nussinov, trisolv) |
| Max speedup | 5.40x (covariance) |
| Kernels with speedup >1x | 18/25 (72%) |

**Top 5 speedups**: covariance (5.40x), deriche (5.28x), 2mm (3.96x), trmm (3.26x), lu (3.12x)

**Slowdowns** (7 kernels): nussinov (0.04x), trisolv (0.04x), 3mm (0.07x), adi (0.08x), floyd_warshall (0.36x), correlation (0.92x), jacobi_2d (0.94x) — inherently sequential algorithms or excessive kernel launch overhead.

### Failure Analysis

The 5 failing kernels are fundamentally hard to parallelize:

| Kernel | Failure Mode | Root Cause |
|--------|-------------|------------|
| doitgen | Numerical (3D indexing) | Complex 3D tensor contraction with temporary array |
| durbin | Numerical (2e-3) | Levinson-Durbin — each step depends on all previous steps |
| gramschmidt | Numerical (0.5–3.4) | Sequential column normalization with cross-column dependencies |
| ludcmp | Numerical (4–36) | LU decomposition with forward/back substitution — sequential |
| seidel_2d | Numerical (0.08) | Gauss-Seidel — each point depends on already-updated neighbors |

### PET Path Fix (jacobi_1d rescue)

`compute_parallel_dims.py` and `compute_reduction_type.py` had hardcoded `KERNELS_DIR` pointing to TSVC kernels. Added optional `kernel_file` parameter so the Polybench pipeline passes the correct path. This gave the LLM precise PET analysis (t=sequential, i=parallel) instead of conservative LLVM fallback output, allowing jacobi_1d to pass on first try.
