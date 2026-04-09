#!/usr/bin/env python3
"""Legacy pattern-specific prompt builder for PolyBench/C kernels.
Preserved for reference and comparison. The active pipeline uses
kernel_analysis.py (unified analysis module) instead."""

import os
import re
import sys
from typing import Optional, Dict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "utilities"))
sys.path.insert(0, "/home/qinxiao/workspace/compiler-guided-triton-gen/analysis")

from polybench_functions_db import POLYBENCH_FUNCTIONS

# Import analysis modules
try:
    from compute_war_dependences import analyze_kernel_war
    HAS_WAR_ANALYSIS = True
except ImportError:
    HAS_WAR_ANALYSIS = False
    analyze_kernel_war = None

try:
    from compute_parallel_dims import analyze_kernel_parallelization
    HAS_PARDIMS_ANALYSIS = True
except ImportError:
    HAS_PARDIMS_ANALYSIS = False

try:
    from compute_scalar_expansion import analyze_kernel_scalar_expansion, format_scalar_expansion_for_prompt
    HAS_SCALAR_EXPANSION = True
except ImportError:
    HAS_SCALAR_EXPANSION = False

try:
    from compute_reduction_type import analyze_kernel_reduction, build_reduction_instructions
    HAS_REDUCTION = True
except ImportError:
    HAS_REDUCTION = False

try:
    from compute_gpu_parallelization_strategy import analyze_kernel_gpu_strategy, build_gpu_strategy_instructions
    HAS_GPU_STRATEGY = True
except ImportError:
    HAS_GPU_STRATEGY = False

try:
    from llvm_fallback_adapters import (
        llvm_war_fallback, llvm_overwrite_fallback,
        llvm_stream_compaction_fallback, llvm_parallel_dims_fallback,
        llvm_scalar_expansion_fallback, try_with_llvm_fallback,
        enhance_war_with_llvm_vectors
    )
    HAS_LLVM_FALLBACK = True
except ImportError:
    HAS_LLVM_FALLBACK = False

POLYBENCH_KERNELS_DIR = "/home/qinxiao/workspace/compiler-guided-triton-gen/analysis/kernels_polybench"
ENABLE_ANALYSIS = True

def load_war_analysis(kernel_name: str) -> Optional[dict]:
    """Load WAR analysis with LLVM fallback and direction vector enhancement."""
    kernel_file = os.path.join(POLYBENCH_KERNELS_DIR, f"{kernel_name}.c")
    if not os.path.exists(kernel_file):
        return None

    pet_result = None
    if HAS_WAR_ANALYSIS and analyze_kernel_war:
        try:
            pet_result = analyze_kernel_war(kernel_file)
        except Exception:
            pass

    # Enhance PET result with LLVM direction vectors for loop-level scoping
    if pet_result and not pet_result.get('parallelization_safe', True) and HAS_LLVM_FALLBACK:
        try:
            enhanced = enhance_war_with_llvm_vectors(kernel_file, pet_result)
            if enhanced:
                return enhanced
        except Exception:
            pass

    if pet_result is not None:
        return pet_result

    # Full LLVM fallback if PET failed entirely
    if HAS_LLVM_FALLBACK:
        try:
            return llvm_war_fallback(kernel_file)
        except Exception:
            pass
    return None


def load_parallelization_analysis(kernel_name: str) -> Optional[dict]:
    """Load parallelization analysis with LLVM fallback."""
    kernel_file = os.path.join(POLYBENCH_KERNELS_DIR, f"{kernel_name}.c")
    if HAS_PARDIMS_ANALYSIS and analyze_kernel_parallelization:
        try:
            result = analyze_kernel_parallelization(kernel_name, kernel_file=kernel_file)
            if result is not None:
                return result
        except Exception:
            pass

    if HAS_LLVM_FALLBACK:
        try:
            return llvm_parallel_dims_fallback(kernel_file)
        except Exception:
            pass
    return None


def load_scalar_expansion_analysis(kernel_name: str) -> Optional[dict]:
    """Load scalar expansion analysis with LLVM fallback."""
    kernel_file = os.path.join(POLYBENCH_KERNELS_DIR, f"{kernel_name}.c")

    if HAS_SCALAR_EXPANSION and analyze_kernel_scalar_expansion:
        try:
            result = analyze_kernel_scalar_expansion(kernel_file)
            if result is not None:
                return result
        except Exception:
            pass

    if HAS_LLVM_FALLBACK:
        try:
            return llvm_scalar_expansion_fallback(kernel_file)
        except Exception:
            pass
    return None


def load_reduction_analysis(kernel_name: str) -> Optional[dict]:
    """Load reduction analysis."""
    if not HAS_REDUCTION or analyze_kernel_reduction is None:
        return None
    kernel_file = os.path.join(POLYBENCH_KERNELS_DIR, f"{kernel_name}.c")
    try:
        return analyze_kernel_reduction(kernel_name, kernel_file=kernel_file)
    except Exception:
        return None



def build_polybench_prompt_legacy(kernel_name: str, func_spec: dict) -> str:
    """Build the prompt for Polybench kernel Triton generation. (Legacy pattern-specific version)"""
    source = get_kernel_source(kernel_name)
    if not source:
        raise ValueError(f"Could not read kernel source for {kernel_name}")

    params = get_kernel_params(kernel_name)
    arrays = func_spec.get('arrays', {})
    scalar_params = func_spec.get('scalar_params', {})
    loop_code = func_spec.get('loop_code', '')
    has_2d = func_spec.get('has_2d_arrays', False)
    has_3d = func_spec.get('has_3d_arrays', False)

    # Build array info section
    array_lines = []
    for arr_name, mode in sorted(arrays.items()):
        mode_str = {'r': 'read-only', 'w': 'write-only', 'rw': 'read-write', 'temp': 'temporary scratch (read-write, not checked for correctness)'}[mode]
        array_lines.append(f"- `{arr_name}`: {mode_str}")
    array_info = "\n".join(array_lines)

    # Build dimension info
    dim_lines = []
    for param_name, param_value in sorted(params.items()):
        dim_lines.append(f"- `{param_name}` = {param_value}")
    dim_info = "\n".join(dim_lines)

    # Build function signature
    sig_parts = []
    for arr_name in sorted(arrays.keys()):
        sig_parts.append(arr_name)
    for sp in sorted(scalar_params.keys()):
        sig_parts.append(sp)
    # Add dimension parameters
    for p in sorted(params.keys()):
        if p not in scalar_params:
            sig_parts.append(p)
    exact_sig = ", ".join(sig_parts)

    # Build valid Python identifier for function names
    func_id = kernel_name
    if func_id[0].isdigit():
        func_id = "k" + func_id

    # Load analysis results (skip if analysis disabled)
    analysis_sections = []

    if ENABLE_ANALYSIS:
        war_result = load_war_analysis(kernel_name)
        par_result = load_parallelization_analysis(kernel_name)
        # Fix: PET's compute_parallel_dims doesn't set has_2d_arrays (returns None).
        # Propagate from func_spec so 2D kernels like 3mm take the correct path.
        if par_result and par_result.get('has_2d_arrays') is None:
            par_result['has_2d_arrays'] = has_2d
        scalar_exp_result = load_scalar_expansion_analysis(kernel_name)
        reduction_result = load_reduction_analysis(kernel_name)

        # Forward/backward substitution override: when WAR shows read array[(j)]
        # vs write array[(i)] with triangular bounds j < i, parallelizing i is
        # unsafe because x[j] reads values from earlier i-iterations.
        if (war_result and not war_result.get('parallelization_safe', True)
                and par_result and par_result.get('is_triangular')
                and par_result.get('options')):
            tri = par_result.get('triangular_info', {})
            smaller = tri.get('smaller', '')   # e.g. j
            larger = tri.get('larger', '')     # e.g. i
            copies = war_result.get('arrays_needing_copy', [])
            deps = war_result.get('war_dependencies', [])
            for dep in deps:
                desc = dep.get('description', '')
                for arr in copies:
                    # Pattern: Read arr[(j)] conflicts with Write arr[(i)]
                    if (f'Read {arr}[({smaller})]' in desc
                            and f'Write {arr}[({larger})]' in desc):
                        for opt in par_result['options']:
                            if opt['parallel_dim'] == larger and opt.get('valid'):
                                opt['valid'] = False
                                opt['issues'].append(
                                    f"Forward substitution: `{arr}[{smaller}]` reads values "
                                    f"from earlier `{larger}` iterations ({smaller} < {larger}). "
                                    f"Parallelizing `{larger}` causes reads of stale values."
                                )

        if war_result and not war_result.get('parallelization_safe', True):
            copies = war_result.get('arrays_needing_copy', [])
            deps = war_result.get('war_dependencies', [])
            loop_scoping = war_result.get('loop_level_scoping')

            # Check if this is single-array triangular (lu/trisolv type) early
            # to customize the WAR header
            _sat_check = (
                not loop_scoping
                and len(copies) == 1
                and par_result and par_result.get('is_triangular')
                and par_result.get('options')
            )
            section = "\n## WAR (Write-After-Read) Dependencies\n\n"
            if _sat_check:
                section += "**Note**: This kernel has WAR (Write-After-Read) dependencies.\n"
            else:
                section += "**Note**: This kernel has WAR (Write-After-Read) dependencies.\n"
                section += "If you split the computation into **separate Triton kernels** launched sequentially, "
                section += "kernel launch barriers handle these dependencies naturally — no cloning needed.\n"
                section += "Cloning is only needed if reads and writes to the same array happen **within a single kernel**.\n"
                section += "\n**Minimize kernel launches**: Fuse compatible phases into a single kernel "
                section += "when they operate on independent data within each thread. For example, "
                section += "a forward sweep and backward sweep on the same row can share a kernel.\n"

            if loop_scoping:
                # Enhanced format with loop-level scoping
                loop_vars = war_result.get('loop_vars', [])
                section += f"\n**Loop variables** (outer to inner): {', '.join(loop_vars)}\n"
                for arr in copies:
                    scoping = loop_scoping.get(arr, {})
                    carried = scoping.get('carried_by_loops', loop_vars)
                    safe = scoping.get('safe_to_parallelize_loops', [])
                    section += f"\n**Array `{arr}`**: WAR carried by loop(s) `{', '.join(carried)}`\n"
                    seq_ctx = scoping.get('sequential_context_loops', [])
                    for var in loop_vars:
                        if var in safe:
                            section += f"- Parallelizing `{var}`: SAFE (no copy needed for `{arr}`)\n"
                        elif var in carried:
                            section += f"- Parallelizing `{var}`: REQUIRES `{arr}_copy = {arr}.clone()`\n"
                        elif var in seq_ctx:
                            section += f"- Loop `{var}`: sequential context (not analyzed for WAR)\n"
            else:
                # Original format without scoping
                # Check if this is a single-array triangular kernel with 1 valid dim
                # (e.g., lu: WAR on A, j VALID, i INVALID cross-phase)
                # In this case, suppress clone/split guidance — it causes the LLM
                # to split into separate phase kernels, doubling launch overhead.
                _is_single_arr_tri = (
                    len(copies) == 1
                    and par_result and par_result.get('is_triangular')
                    and par_result.get('options')
                )
                _par_valid = [o for o in par_result['options'] if o.get('valid')] if par_result and par_result.get('options') else []
                _par_invalid_cp = [o for o in par_result['options']
                                   if not o.get('valid') and any('Cross-phase' in iss for iss in o.get('issues', []))
                                   ] if par_result and par_result.get('options') else []
                if _is_single_arr_tri and len(_par_valid) == 1 and _par_invalid_cp:
                    # Single-array triangular with 1 valid dim: give structural guidance
                    seq_dim = _par_invalid_cp[0]['parallel_dim']
                    par_dim = _par_valid[0]['parallel_dim']
                    section += f"\n**Arrays with WAR dependencies**: {', '.join(copies)}\n"
                    section += f"\n**Recommended structure**: Use `grid=(1,)` with the `{seq_dim}` loop "
                    section += f"**INSIDE** the kernel. This puts the entire computation in a single "
                    section += "kernel launch, avoiding N separate kernel launch overheads.\n"
                    section += "```python\n"
                    section += f"kernel[grid=(1,)]({copies[0]}, N)  # ONE kernel launch total\n"
                    section += f"# Inside kernel: for {seq_dim} in range(N): vectorize {par_dim}\n"
                    section += "```\n"
                    section += f"With `grid=(1,)` (single thread block), the sequential `{seq_dim}` loop is correct — "
                    section += "there are no cross-CTA race conditions. Process BOTH phases "
                    section += f"(all `{par_dim}` ranges) within the `{seq_dim}` loop. "
                    section += "No cloning needed.\n"
                    section += f"\n**CRITICAL: Vectorize `{par_dim}`**: Use `BLOCK_SIZE = min(triton.next_power_of_2(N), 128)` "
                    section += f"and `{par_dim}_offsets = tl.arange(0, BLOCK_SIZE)` to process ALL `{par_dim}` "
                    section += "values simultaneously. Do NOT use a scalar `for` loop over "
                    section += f"`{par_dim}` — use **masked vector operations** instead. "
                    section += f"For inner-product loops (e.g., `for k`), load `{copies[0]}[{seq_dim}][k]` "
                    section += f"as a scalar and `{copies[0]}[k][{par_dim}_offsets]` as a vector, then do "
                    section += "vectorized multiply-subtract across all columns simultaneously.\n"
                elif _is_single_arr_tri and not _par_valid:
                    # Single-array triangular with NO valid dims (e.g., forward substitution):
                    # fully sequential kernel, no cloning needed
                    section += f"\n**Arrays with WAR dependencies**: {', '.join(copies)}\n"
                    section += "\nThis kernel is **inherently sequential** — no dimension can be safely "
                    section += "parallelized. Use `grid=(1,)` with sequential loops. "
                    section += "No cloning needed since execution is sequential.\n"
                else:
                    # Check if there are NO valid parallelization options.
                    # If so, recommend sequential grid=(1,) instead of cloning
                    # (cloning is pointless when nothing can be parallelized,
                    # and it confuses the LLM into attempting broken parallelism).
                    _has_valid_par = (
                        par_result and par_result.get('options')
                        and any(o.get('valid') for o in par_result['options'])
                    )
                    if copies:
                        section += f"\n**Arrays with WAR dependencies**: {', '.join(copies)}\n"
                    for dep in deps[:5]:
                        section += f"- {dep.get('description', '')}\n"
                    if not _has_valid_par:
                        # No parallelizable dimensions — recommend sequential processing
                        section += "\nThis kernel has **no safely parallelizable dimensions**. "
                        section += "Use `grid=(1,)` with sequential loops inside the kernel. "
                        section += "No cloning needed since execution is sequential within a single CTA.\n"
                    else:
                        if copies:
                            section += "\n**If using a single kernel**: Create read-only copies before the parallel region:\n"
                            section += "```python\n"
                            for arr in copies:
                                section += f"{arr}_copy = {arr}.clone()  # Read from copy, write to original\n"
                            section += "```\n"
                            section += "**If using separate kernels**: No cloning needed — launch one kernel per phase.\n"
                        # Soften clone guidance when ParDims shows ≥2 valid dims with sequential outer
                        if par_result and par_result.get('options'):
                            valid = [o for o in par_result['options'] if o.get('valid')]
                            invalid = [o for o in par_result['options'] if not o.get('valid')]
                            if len(valid) >= 2 and invalid:
                                seq_dim = invalid[0]['parallel_dim']
                                section += f"\n**Note**: Both spatial dimensions are safe to parallelize (see below). "
                                section += f"When `{seq_dim}` is iterated sequentially in Python host code "
                                section += "with separate kernel launches, cloning is likely unnecessary — "
                                section += "each thread writes to a unique `(i,j)` location.\n"

            analysis_sections.append(section)

        if par_result and par_result.get('options'):
            valid_opts = [o for o in par_result['options'] if o['valid']]
            # Only include parallelization section if at least one option is valid
            if valid_opts:
                section = "\n## Parallelization Analysis\n\n"
                section += f"**Loop dimensions**: {par_result.get('dims', [])}\n"
                if par_result.get('is_triangular'):
                    tri = par_result['triangular_info']
                    section += f"**Triangular bounds**: {tri.get('smaller', '?')} < {tri.get('larger', '?')}\n"
                # Detect 1D stencil pattern: 1 valid dim, not 2D, cross-phase with 2+ write arrays
                _is_1d_stencil = (
                    len(valid_opts) == 1
                    and not par_result.get('has_2d_arrays', False)
                    and any(not o['valid'] and any('Cross-phase' in iss for iss in o.get('issues', []))
                            for o in par_result['options'])
                )
                for opt in par_result['options']:
                    valid = "VALID" if opt['valid'] else "INVALID"
                    if opt['valid'] and _is_1d_stencil:
                        valid = "VALID (vectorize within single CTA — MUST use grid=(1,), see below)"
                    section += f"\n- Parallelize `{opt['parallel_dim']}`, sequential `{opt['sequential_dim']}`: {valid}\n"
                    for issue in opt.get('issues', []):
                        section += f"  - {issue}\n"
                    # When valid and sequential dim is a reduction, recommend BLOCK_SIZE >= 128
                    if opt['valid']:
                        seq_dim = opt.get('sequential_dim', '')
                        _seq_is_reduction = any(
                            not o['valid'] and any('Write conflict' in iss for iss in o.get('issues', []))
                            for o in par_result['options'] if o['parallel_dim'] == seq_dim
                        )
                        if _seq_is_reduction:
                            section += (f"  - **Reduction on `{seq_dim}`**: vectorize with `tl.arange()` "
                                        f"and reduce with `tl.sum()`. Use **BLOCK_SIZE = 64** for the "
                                        f"`{seq_dim}` dimension to balance throughput and occupancy.\n")
                # Fix 7: When 1 valid dim + 2+ sequential dims, fuse sequential dims into grid
                # and vectorize the parallel dim inside each program.
                # IMPORTANT: Do NOT put the parallel dim in the grid — when the main array
                # is read-write (in-place update with temp), different p-blocks for the same
                # (r,q) would race: one writes A[r][q][p1] while another reads A[r][q][s=p1].
                all_dims = par_result.get('dims', [])
                valid_dim_names_set = {o['parallel_dim'] for o in valid_opts}
                # Exclude timestep/sequential-context dims from fusable sequential dims
                # (timesteps must stay in Python host code, not fused into grid)
                _seq_context_dim_set = {
                    o['parallel_dim'] for o in par_result['options']
                    if not o['valid'] and any('sequential context' in iss.lower()
                                              for iss in o.get('issues', []))
                }
                seq_dim_names = [d for d in all_dims
                                 if d not in valid_dim_names_set
                                 and d not in _seq_context_dim_set]
                n_seq_dims = len(seq_dim_names)
                if len(valid_opts) == 1 and n_seq_dims >= 2:
                    par_dim_name = valid_opts[0]['parallel_dim']
                    seq_upper = ' * '.join(d.upper() for d in seq_dim_names)
                    seq_names_str = ', '.join(seq_dim_names)
                    # Find read-write arrays that need cloning
                    rw_arrs = sorted([a for a, m in arrays.items() if m == 'rw'])
                    section += f"\n**CRITICAL: Grid = sequential dimensions ONLY. Vectorize `{par_dim_name}` inside.**\n"
                    section += f"With {n_seq_dims + 1} loop dimensions and only `{par_dim_name}` parallelizable, "
                    section += f"encode ONLY the sequential dims (`{seq_names_str}`) in the grid.\n"
                    if rw_arrs:
                        section += f"\n**MANDATORY: Clone read-write arrays in the wrapper BEFORE launching the kernel:**\n"
                        section += f"```python\n"
                        section += f"def wrapper({', '.join(sorted(arrays.keys()))}, ...):\n"
                        for arr in rw_arrs:
                            section += f"    {arr}_copy = {arr}.clone()  # Read from copy\n"
                        section += f"    BLOCK = min(triton.next_power_of_2({par_dim_name.upper()}), 128)\n"
                        section += f"    grid = ({seq_upper},)\n"
                        section += f"    kernel[grid]({', '.join(rw_arrs[0] + ', ' + rw_arrs[0] + '_copy' if len(rw_arrs) == 1 else '...')}, ...)\n"
                        section += f"```\n"
                        section += f"The kernel receives BOTH the output array and the read-only copy:\n"
                        section += f"```python\n"
                        section += f"@triton.jit\n"
                        section += f"def kernel({rw_arrs[0]}_out, {rw_arrs[0]}_in, ..., BLOCK: tl.constexpr):\n"
                    else:
                        section += f"```python\n"
                        section += f"@triton.jit\n"
                        section += f"def kernel(...):\n"
                    section += f"    pid = tl.program_id(0)\n"
                    if n_seq_dims == 2:
                        section += f"    {seq_dim_names[1]} = pid % {seq_dim_names[1].upper()}\n"
                        section += f"    {seq_dim_names[0]} = pid // {seq_dim_names[1].upper()}\n"
                    else:
                        for sd in seq_dim_names:
                            section += f"    # decode {sd} from pid\n"
                    section += f"    {par_dim_name}_offsets = tl.arange(0, BLOCK)\n"
                    section += f"    mask = {par_dim_name}_offsets < {par_dim_name.upper()}\n"
                    section += f"    acc = tl.zeros([BLOCK], dtype=tl.float32)\n"
                    section += f"    for s in range({par_dim_name.upper()}):\n"
                    if rw_arrs:
                        section += f"        val = tl.load({rw_arrs[0]}_in + ...)  # scalar from READ-ONLY copy\n"
                    else:
                        section += f"        val = tl.load(...)  # scalar from input\n"
                    section += f"        vec = tl.load(...)  # vector[BLOCK] from weight matrix\n"
                    section += f"        acc += val * vec\n"
                    if rw_arrs:
                        section += f"    tl.store({rw_arrs[0]}_out + ..., acc, mask=mask)  # Write to ORIGINAL\n"
                    else:
                        section += f"    tl.store(output + ..., acc, mask=mask)\n"
                    section += f"```\n"
                    section += f"This launches ONE kernel call, not nested Python for-loops. "
                    if rw_arrs:
                        section += f"The clone ensures reads from `{rw_arrs[0]}_in` see the original values "
                        section += f"even as other programs write to `{rw_arrs[0]}_out`.\n"
                    else:
                        section += f"Each program handles ALL `{par_dim_name}` values for its (`{seq_names_str}`) pair.\n"
                    section += f"\n**Temporary accumulators** (like `sum`): Do NOT allocate as a global tensor. "
                    section += f"Use a **register-local vector** (`tl.zeros`) inside each program instance. "
                    section += f"A global `sum` tensor shared across parallel programs causes **race conditions**.\n"
                # When both dims are freely parallelizable, recommend 2D grid
                # Also handle N-D analysis where a single option has comma-separated dims
                _multi_dim_opt = (len(valid_opts) == 1
                                  and ',' in valid_opts[0].get('parallel_dim', ''))
                if len(valid_opts) >= 2 or _multi_dim_opt:
                    if _multi_dim_opt:
                        valid_dim_names = [d.strip() for d in valid_opts[0]['parallel_dim'].split(',')][:2]
                    else:
                        valid_dim_names = [o['parallel_dim'] for o in valid_opts[:2]]
                    section += f"\n**Both `{valid_dim_names[0]}` and `{valid_dim_names[1]}` are freely parallelizable.** "
                    section += "Use a 2D grid to parallelize both simultaneously for best GPU occupancy.\n"
                    # Detect GEMM accumulation pattern for freely-parallelizable 2D kernels
                    # Match: C[i][j] += A[i][k] * B[k][j]  OR  C[i][j] += alpha * A[i][k] * B[k][j]
                    _matmul_re = (r'\w+\[.*?\]\[.*?\]\s*\+=\s*'
                                  r'(?:\w+\s*\*\s*)?'  # optional scalar multiplier (alpha *)
                                  r'\w+\[.*?\]\[.*?\]\s*\*\s*\w+\[.*?\]\[.*?\]')
                    if has_2d and re.search(_matmul_re, loop_code):
                        # Check if problem sizes are large enough for 2D tiled matmul
                        _dim_values = [params.get(d, 0) for d in params]
                        _max_dim = max(_dim_values) if _dim_values else 0
                        if _max_dim >= 128:
                            section += ("\n**Matrix Multiply Pattern**: Use `tl.dot()` for the inner reduction loop.\n"
                                        "**CRITICAL**: Do NOT use scalar k-loop accumulation. Use tiled matmul:\n"
                                        "```python\n"
                                        "BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 32\n"
                                        "grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))\n"
                                        "@triton.jit\n"
                                        "def matmul_kernel(C_ptr, A_ptr, B_ptr, M, N, K, ...):\n"
                                        "    pid_m = tl.program_id(0)\n"
                                        "    pid_n = tl.program_id(1)\n"
                                        "    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)\n"
                                        "    for k in range(0, K, BLOCK_K):\n"
                                        "        a = tl.load(A_ptr + ...)  # [BLOCK_M, BLOCK_K] tile\n"
                                        "        b = tl.load(B_ptr + ...)  # [BLOCK_K, BLOCK_N] tile\n"
                                        "        acc += tl.dot(a, b)\n"
                                        "    # If scalar multiplier (alpha): acc = alpha * acc\n"
                                        "    tl.store(C_ptr + ..., acc, ...)\n"
                                        "```\n"
                                        "Use BLOCK_M=BLOCK_N=BLOCK_K=32 (not 16) for sufficient GPU occupancy.\n")
                        else:
                            # Small matrices: 1D row-parallel is faster than 2D tiled matmul
                            # because grid=(M,) gives more CTAs than grid=(M/32, N/32)
                            section += ("\n**Matrix Multiply Pattern (small problem size)**:\n"
                                        "With dimensions < 128, use a **1D row-parallel** strategy for maximum GPU occupancy:\n"
                                        "```python\n"
                                        "grid = (M,)  # one CTA per output row\n"
                                        "@triton.jit\n"
                                        "def matmul_kernel(C_ptr, A_ptr, B_ptr, M, N, K, alpha, beta, BLOCK_N: tl.constexpr):\n"
                                        "    row = tl.program_id(0)\n"
                                        "    # Scale existing C row by beta\n"
                                        "    col_offs = tl.arange(0, BLOCK_N)\n"
                                        "    mask = col_offs < N\n"
                                        "    c_row = tl.load(C_ptr + row * N + col_offs, mask=mask)\n"
                                        "    c_row = beta * c_row\n"
                                        "    # Accumulate A[row,:] @ B into c_row\n"
                                        "    for k in range(K):\n"
                                        "        a_val = tl.load(A_ptr + row * K + k)\n"
                                        "        b_row = tl.load(B_ptr + k * N + col_offs, mask=mask)\n"
                                        "        c_row += alpha * a_val * b_row\n"
                                        "    tl.store(C_ptr + row * N + col_offs, c_row, mask=mask)\n"
                                        "```\n"
                                        "Use `BLOCK_N = min(triton.next_power_of_2(N), 128)` to cover the row in chunks for good occupancy.\n"
                                        "This gives M CTAs (one per row) instead of ceil(M/32)*ceil(N/32) with 2D tiling.\n")
                    # Multi-phase stencil kernels: sequential timestep + 2+ valid spatial dims
                    # Detect sequential timestep dim: either from options (LLVM) or from
                    # dims list minus options (PET N-D, where t is excluded entirely)
                    seq_ctx_dims = [
                        o for o in par_result['options']
                        if not o['valid'] and any('sequential context' in iss.lower()
                                                  for iss in o.get('issues', []))
                    ]
                    t_dim = None
                    if seq_ctx_dims:
                        t_dim = seq_ctx_dims[0]['parallel_dim']
                    else:
                        # N-D analysis: t may be absent from options but present in dims
                        all_opt_dims = set()
                        for o in par_result['options']:
                            for d in o['parallel_dim'].split(','):
                                all_opt_dims.add(d.strip())
                        all_dims = par_result.get('dims', [])
                        missing_dims = [d for d in all_dims if d not in all_opt_dims]
                        if missing_dims:
                            t_dim = missing_dims[0]
                    # Detect n_write_arrays from LLVM result, self_deps, or c_code
                    _n_write = par_result.get('n_write_arrays', 0)
                    if _n_write == 0:
                        _n_write = len(set(d['array'] for d in par_result.get('self_dependencies', [])
                                          if 'write_expr' in d))
                    if _n_write == 0:
                        import re as _re2
                        _c_code = par_result.get('c_code', '')
                        _n_write = len(set(_re2.findall(r'^\s*(\w+)\s*\[', _c_code, _re2.MULTILINE)))
                    if t_dim and _n_write >= 2:
                        # Count total valid spatial dimensions
                        if _multi_dim_opt:
                            _total_valid_dims = len([d.strip() for d in valid_opts[0]['parallel_dim'].split(',')])
                        else:
                            _total_valid_dims = len(valid_opts)

                        # 2D+ stencils: ALWAYS use multi-CTA with Python t-loop.
                        # Kernel launch between phases acts as synchronization barrier.
                        # Even at small N, N² elements justify multi-CTA parallelism
                        # (a single CTA wastes 81/82 SMs on RTX 3090).
                        # 1D stencils (handled separately below) use grid=(1,) since
                        # N elements don't justify multi-CTA overhead.
                        if _total_valid_dims >= 2:
                            section += f"\n**CRITICAL: Timestep/phase structure**: The `{t_dim}` loop must be in "
                            section += "**Python host code**, NOT inside the Triton kernel.\n"
                            section += f"Do NOT put `for {t_dim} in range(...)` inside the Triton kernel — there is no "
                            section += "global synchronization between timesteps within a single kernel launch, "
                            section += "which causes **race conditions** on shared arrays.\n"
                            section += "\n**Use at most 2 kernels per timestep**. Fuse independent phases "
                            section += "into a single kernel when they write to different arrays.\n"
                            section += "\n**Parallelize ALL spatial dimensions**: Flatten all spatial dimensions "
                            section += "into a single 1D index. Each CTA processes a block of elements from the "
                            section += "flattened space, recovering multi-dim coordinates from the linear index.\n"
                            section += "\n**Use BLOCK_SIZE = 128** (not larger) to maximize the number of CTAs "
                            section += "and GPU occupancy.\n"
                            section += "```python\n"
                            section += f"BLOCK_SIZE = 128\n"
                            section += f"total_elements = N_DIM0 * N_DIM1  # all spatial dims (add more if 3D+)\n"
                            section += f"grid = (triton.cdiv(total_elements, BLOCK_SIZE),)\n"
                            section += f"for {t_dim} in range(TSTEPS):\n"
                            section += f"    phase1_kernel[grid](...)  # Phase 1\n"
                            section += f"    phase2_kernel[grid](...)  # Phase 2 (kernel launch barrier = sync)\n"
                            section += "```\n"
                            section += "Inside the kernel, recover coordinates from the flat index:\n"
                            section += "```python\n"
                            section += "flat_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)\n"
                            section += "j = flat_idx % N_COLS + 1  # +1 to skip boundary\n"
                            section += "i = flat_idx // N_COLS + 1\n"
                            section += "mask = (i < N - 1) & (j < N - 1)  # boundary check\n"
                            section += "```\n"
                            section += "\n**Do NOT use `grid=(1,)`** — a single CTA wastes GPU parallelism. "
                            section += "Kernel launches between phases provide synchronization, so "
                            section += "multi-CTA is both correct and fast.\n"
                # Fix 12: 1 valid spatial dim + timestep + recurrence on other dim
                # (e.g., ADI: i parallel, j has recurrence, t timestep)
                # Detect timestep dim for 1-valid-dim case
                _seq_ctx_for_sweep = [
                    o for o in par_result['options']
                    if not o['valid'] and any('sequential context' in iss.lower()
                                              for iss in o.get('issues', []))
                ]
                _recurrence_dims = [
                    o for o in par_result['options']
                    if not o['valid'] and any('Dependencies carried' in iss
                                              for iss in o.get('issues', []))
                ]
                if (len(valid_opts) == 1 and _seq_ctx_for_sweep and _recurrence_dims
                        and par_result.get('n_write_arrays', 0) >= 2):
                    _t_dim = _seq_ctx_for_sweep[0]['parallel_dim']
                    par_dim = valid_opts[0]['parallel_dim']
                    section += f"\n**Tridiagonal/IIR sweep pattern**: `{par_dim}` is parallelizable, "
                    section += f"but the other spatial dimension has sequential recurrence "
                    section += f"(e.g., `x[j]` depends on `x[j-1]`).\n\n"
                    section += f"**Use `grid=({par_dim.upper()}-2,)` — ONE CTA per {par_dim}-index** "
                    section += f"(NOT `grid=(cdiv(N, BLOCK),)` with a for-loop inside each CTA).\n"
                    section += f"Each CTA runs the full sequential sweep for its row/column.\n\n"
                    section += "```python\n"
                    section += f"for {_t_dim} in range(TSTEPS):  # timestep loop in Python\n"
                    section += f"    sweep1_kernel[({par_dim.upper()}-2,)](...)\n"
                    section += f"    sweep2_kernel[({par_dim.upper()}-2,)](...)\n"
                    section += "```\n\n"
                    section += "Inside each kernel:\n"
                    section += "```python\n"
                    section += "@triton.jit\n"
                    section += f"def sweep_kernel(...):\n"
                    section += f"    {par_dim} = tl.program_id(0) + 1  # One CTA per {par_dim}, skip boundary\n"
                    section += f"    # Forward sweep: scalar for-loop over recurrence dim\n"
                    section += f"    for j in range(1, N-1):\n"
                    section += f"        # Tridiagonal/IIR computation using scalar loads\n"
                    section += f"        ...\n"
                    section += f"    # Backward sweep: scalar for-loop in reverse\n"
                    section += f"    for j in range(N-2, 0, -1):\n"
                    section += f"        ...\n"
                    section += "```\n\n"
                    section += "**IMPORTANT**: Do NOT use BLOCK_SIZE with a for-loop over blocks "
                    section += "inside each CTA. Each CTA handles exactly ONE row/column. "
                    section += f"`grid=({par_dim.upper()}-2,)` gives {par_dim.upper()}-2 CTAs "
                    section += "for maximum GPU occupancy.\n"

                # When some dims are invalid due to cross-phase deps, provide
                # targeted guidance based on whether phases write distinct arrays
                cross_phase_invalids = [
                    o for o in par_result['options']
                    if not o['valid'] and any('Cross-phase' in iss for iss in o.get('issues', []))
                ]
                if cross_phase_invalids:
                    seq_dims = [o['parallel_dim'] for o in cross_phase_invalids]
                    # Count distinct write arrays
                    _cp_n_write = len(set(d['array'] for d in par_result.get('self_dependencies', [])
                                         if 'write_expr' in d))
                    # Check if invalid dim has write conflicts (truly non-parallelizable
                    # even within individual phases) vs only cross-phase (safe within
                    # each phase when launched as separate kernels)
                    _has_write_conflict = any(
                        'Write conflict' in iss
                        for o in cross_phase_invalids
                        for iss in o.get('issues', [])
                    )
                    if _cp_n_write >= 2 and not _has_write_conflict:
                        # Independent phases writing different arrays, no write conflicts
                        if len(valid_opts) == 1 and not par_result.get('has_2d_arrays', False):
                            # 1D stencil (e.g., jacobi_1d): grid=(1,) with barriers
                            # N elements don't justify multi-CTA overhead
                            par_dim = valid_opts[0]['parallel_dim']
                            section += f"\n**CRITICAL**: Use `grid=(1,)` with the `{seq_dims[0]}` loop and ALL phases "
                            section += "INSIDE a single kernel. Insert `tl.debug_barrier()` between phases "
                            section += "to flush stores before the next phase reads the same array.\n"
                            section += f"\n**MANDATORY REQUIREMENTS:**\n"
                            section += f"- `grid = (1,)` — a SINGLE CTA (NOT `triton.cdiv(N, BLOCK)`!)\n"
                            section += f"- `BLOCK = triton.next_power_of_2(N)` — process ALL elements in one vector, NO for-loop over blocks\n"
                            section += f"- `tl.debug_barrier()` between EVERY phase boundary — without it, loads read stale cache data\n"
                            section += f"\n**Copy this COMPLETE implementation (kernel + wrapper):**\n"
                            section += "```python\n"
                            section += f"@triton.jit\n"
                            section += f"def {func_id}_kernel(A_ptr, B_ptr, N: tl.constexpr, {seq_dims[0].upper()}: tl.constexpr, BLOCK: tl.constexpr):\n"
                            section += f"    offsets = tl.arange(0, BLOCK)\n"
                            section += f"    mask = (offsets >= 1) & (offsets < N - 1)\n"
                            section += f"    for {seq_dims[0]} in range({seq_dims[0].upper()}):\n"
                            section += f"        tl.debug_barrier()  # REQUIRED: flush previous stores\n"
                            section += f"        a_l = tl.load(A_ptr + offsets - 1, mask=mask)\n"
                            section += f"        a_c = tl.load(A_ptr + offsets, mask=mask)\n"
                            section += f"        a_r = tl.load(A_ptr + offsets + 1, mask=mask)\n"
                            section += f"        tl.store(B_ptr + offsets, 0.33333 * (a_l + a_c + a_r), mask=mask)\n"
                            section += f"        tl.debug_barrier()  # REQUIRED: flush B stores before Phase 2\n"
                            section += f"        b_l = tl.load(B_ptr + offsets - 1, mask=mask)\n"
                            section += f"        b_c = tl.load(B_ptr + offsets, mask=mask)\n"
                            section += f"        b_r = tl.load(B_ptr + offsets + 1, mask=mask)\n"
                            section += f"        tl.store(A_ptr + offsets, 0.33333 * (b_l + b_c + b_r), mask=mask)\n"
                            section += f"\n"
                            section += f"def {func_id}_triton({exact_sig}):\n"
                            section += f"    BLOCK = triton.next_power_of_2(N)\n"
                            section += f"    {func_id}_kernel[(1,)](A, B, N=N, {seq_dims[0].upper()}={seq_dims[0].upper()}, BLOCK=BLOCK)\n"
                            section += "```\n"
                            section += "`tl.debug_barrier()` acts as a memory fence within the CTA, ensuring "
                            section += "vectorized stores are visible to subsequent vectorized loads. "
                            section += "Without it, loads may read stale L1 cache data.\n"
                            section += "\n**WARNING: Do NOT split phases into separate kernel launches.** "
                            section += f"Launching multiple kernels per timestep × {seq_dims[0].upper()} timesteps "
                            section += "creates massive launch overhead. "
                            section += "Keep ALL timesteps and ALL phases inside a SINGLE kernel with `grid=(1,)`. "
                            section += "Use `tl.debug_barrier()` between every phase boundary to ensure "
                            section += "stores from phase 1 are visible to loads in phase 2.\n"
                            section += "\n**WARNING: Do NOT use multi-block grids.** "
                            section += "Multiple CTAs cannot synchronize between phases — CTA 0 may execute Phase 2 "
                            section += "while CTA 1 is still in Phase 1, reading stale values from the boundary. "
                            section += "Only `grid=(1,)` guarantees correctness for stencils with overlapping read/write halos.\n"
                        else:
                            # 2D+ problem (e.g., 3mm: E=A*B, F=C*D, G=E*F)
                            # → split into separate kernels, each CAN parallelize both dims
                            section += f"\n**IMPORTANT**: `{'`, `'.join(seq_dims)}` is INVALID across phases, "
                            section += "but **within each separate kernel**, both dimensions are safe to parallelize. "
                            section += "**Split into separate Triton kernels** per phase, launched sequentially "
                            section += "from Python. Within each kernel, parallelize **BOTH** dimensions "
                            section += "using a 2D grid — kernel launch barriers resolve the cross-phase "
                            section += "dependencies.\n"
                            # Detect GEMM accumulation: out[i][j] += [alpha *] lhs[i][k] * rhs[k][j]
                            _matmul_re2 = (r'\w+\[.*?\]\[.*?\]\s*\+=\s*'
                                           r'(?:\w+\s*\*\s*)?'
                                           r'\w+\[.*?\]\[.*?\]\s*\*\s*\w+\[.*?\]\[.*?\]')
                            if re.search(_matmul_re2, loop_code):
                                section += ("\n**Matrix Multiply Pattern**: Use `tl.dot()` for the inner reduction:\n"
                                            "- Grid: `(cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))` per phase kernel\n"
                                            "- Each CTA: load A[BLOCK_M, BLOCK_K] and B[BLOCK_K, BLOCK_N] tiles, "
                                            "accumulate `acc += tl.dot(a_tile, b_tile)` over k-blocks\n"
                                            "- BLOCK_M=BLOCK_N=BLOCK_K=16 (minimum for tl.dot)\n")
                    elif _cp_n_write >= 2 and _has_write_conflict:
                        # Phases share write locations (truly non-parallelizable outer dim)
                        # (e.g., gramschmidt: k sequential, phases dependent within each k)
                        # Use grid=(1,) with k-loop inside to minimize launch overhead
                        par_dim = valid_opts[0]['parallel_dim']
                        section += f"\n**IMPORTANT**: Use `grid=(1,)` with the `{'`, `'.join(seq_dims)}` loop "
                        section += f"**INSIDE** the kernel — this puts everything in a single kernel launch, "
                        section += "avoiding massive launch overhead.\n"
                        section += "```python\n"
                        section += f"kernel[grid=(1,)](...)  # ONE launch — {seq_dims[0]} loop inside\n"
                        section += "```\n"
                        section += f"With `grid=(1,)`, the sequential `{seq_dims[0]}` loop is correct (single CTA, "
                        section += "no races). Process ALL phases within each iteration:\n"
                        section += f"- Reductions: use `tl.sum()` — do NOT use `tl.atomic_add`\n"
                        section += f"- Per-`{par_dim}` work: use a `for {par_dim}` loop inside the kernel\n"
                        section += "- All column/row operations: vectorize with `tl.arange(0, BLOCK_SIZE)`\n"
                    # For single-write-array (e.g., lu): no special structural guidance —
                    # let the LLM choose the simplest approach (typically per-iteration
                    # kernel launches with vectorized inner dim).
                analysis_sections.append(section)
            else:
                # All options INVALID — check if write_conflict pattern (opposing reductions)
                # suggests loop fission into separate kernels.
                # Only count conflicts on actual arrays (non-empty subscripts like [(i)]),
                # not scalars (write [] = scalar temp variable → scalar expansion issue).
                import re as _re
                write_conflict_opts = []
                for opt in par_result['options']:
                    issues = opt.get('issues', [])
                    array_conflicts = [
                        iss for iss in issues
                        if 'Write conflict' in iss
                        and _re.search(r'write \[[^\]]+\]', iss)  # non-empty subscript = array
                    ]
                    if array_conflicts:
                        write_conflict_opts.append(opt)
                if len(write_conflict_opts) >= 2:
                    dims = par_result.get('dims', [])
                    # Detect shared 2D array: opposing reductions on a common matrix.
                    # Splitting would force one kernel to use strided column access.
                    # Instead, fuse into a single kernel that iterates rows for coalesced access.
                    _has_2d = par_result.get('has_2d_arrays', False)
                    if not _has_2d:
                        # Check from c_code if 2D array indexing exists
                        _c = par_result.get('c_code', '')
                        _has_2d = bool(_re.search(r'\w+\[[\w+]+\]\[[\w+]+\]', _c))

                    section = "\n## Opposing Reductions — Fused Kernel\n\n"
                    section += "**Neither dimension can be parallelized alone** "
                    section += "because the loop body has reductions into different arrays along opposing dimensions:\n"
                    for opt in write_conflict_opts:
                        for iss in opt.get('issues', []):
                            if 'Write conflict' in iss:
                                section += f"- {iss}\n"
                    section += "\n**Fuse both reductions into a SINGLE kernel** "
                    section += "that iterates **rows** of the shared 2D array "
                    section += "for coalesced memory access. Both reductions share the same "
                    section += "row-major loads — splitting would force one kernel into "
                    section += "strided column access.\n\n"
                    section += "**Pattern**:\n"
                    section += "```python\n"
                    section += "@triton.jit\n"
                    section += "def fused_kernel(A_ptr, vec1_ptr, vec2_ptr, out1_ptr, out2_ptr,\n"
                    section += "                 M: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):\n"
                    section += "    offsets = tl.arange(0, BLOCK)\n"
                    section += "    # Accumulator for column-wise output (e.g., s[j])\n"
                    section += "    col_acc = tl.zeros([BLOCK], dtype=tl.float32)\n"
                    section += "    for i in range(M):  # iterate ROWS\n"
                    section += "        mask = offsets < N\n"
                    section += "        a_row = tl.load(A_ptr + i * N + offsets, mask=mask)  # COALESCED row load\n"
                    section += "        v1 = tl.load(vec1_ptr + i)  # scalar\n"
                    section += "        # Row reduction → one output per row\n"
                    section += "        row_sum = tl.sum(a_row * tl.load(vec2_ptr + offsets, mask=mask))\n"
                    section += "        tl.store(out1_ptr + i, row_sum)  # e.g., q[i]\n"
                    section += "        # Column accumulation → vector output\n"
                    section += "        col_acc += v1 * a_row  # e.g., s[j] += r[i] * A[i][j]\n"
                    section += "    # Store column-wise result\n"
                    section += "    tl.store(out2_ptr + offsets, col_acc, mask=offsets < N)\n"
                    section += "grid = (1,)  # single kernel, both reductions fused\n"
                    section += "```\n"
                    section += "\n**Key**: Iterate the ROW dimension (M) as the outer loop. "
                    section += "Vectorize the COLUMN dimension (N) with `tl.arange()`. "
                    section += "This ensures ALL loads from the 2D array are coalesced (stride-1).\n"
                    analysis_sections.append(section)
                elif (any('Cross-phase' in iss
                         for o in par_result['options']
                         for iss in o.get('issues', []))
                      and not analysis_sections
                      and not (scalar_exp_result and scalar_exp_result.get('has_scalar_expansion'))
                      and not (reduction_result and reduction_result.get('is_reduction'))):
                    # Multi-phase PET kernel with no other analysis: different
                    # loop bodies under the outer dim read/write the same arrays.
                    # Each phase IS independently parallelizable — tell the LLM
                    # to split.  Only emit when no WAR/ScalarExp/Reduction
                    # already provides more specific guidance.
                    section = "\n## Multi-Phase Kernel — Split into Separate Triton Kernels\n\n"
                    section += "**Both dimensions appear INVALID when the phases are analyzed together**, "
                    section += "but that is because different loop bodies (phases) share arrays across iterations.\n\n"
                    section += "**Each phase can be parallelized independently.** "
                    section += "Split into **separate Triton kernels** — one per top-level `for` loop — "
                    section += "and parallelize the spatial dimensions within each kernel. "
                    section += "Kernel launch barriers handle cross-phase deps.\n\n"
                    section += ("For each kernel, use `grid=(N,)` with one CTA per row. "
                               "Within each kernel:\n"
                               "- For phases computing row reductions (e.g., `y[i] += A[i][j] * x[j]`): "
                               "parallelize rows with `grid=(N,)`, vectorize columns with `tl.arange()`\n"
                               "- For inner reduction loops: use `tl.sum()` over vectorized products\n"
                               "- Use **BLOCK_SIZE = 64** for column/reduction dimensions\n"
                               "- **Fuse element-wise phases** (e.g., `x[i] = x[i] + z[i]`) into the "
                               "preceding or following reduction kernel to minimize launch count. "
                               "Only split when phases have true data dependencies between them.\n")
                    # Fix 8: When phases look like matrix multiplies, add tl.dot guidance
                    if par_result.get('has_2d_arrays', False):
                        section += "\n**For matrix-multiply phases**: Use 2D tiled grid `grid = (cdiv(M, BM), cdiv(N, BN))` "
                        section += "with `tl.dot()` accumulation. Each block computes a BM×BN output tile by "
                        section += "iterating over the K dimension: `for k in range(0, K, BK): acc += tl.dot(a_tile, b_tile)`.\n"
                        section += "**IMPORTANT**: The host function MUST call these tiled phase kernels. "
                        section += "Do NOT create an additional serial fallback kernel.\n"
                    analysis_sections.append(section)
                elif par_result.get('source') == 'llvm':
                    # LLVM fallback: direction vectors confirm deps carried on all dims.
                    dims = par_result.get('dims', [])
                    section = "\n## Parallelization Warning\n\n"
                    section += "**No dimension is safe to parallelize independently.** "
                    section += "Data dependencies are carried along ALL loop dimensions:\n"
                    for opt in par_result['options']:
                        for iss in opt.get('issues', []):
                            section += f"- `{opt['parallel_dim']}`: {iss}\n"
                    n_write = par_result.get('n_write_arrays', 1)
                    if n_write >= 2:
                        # Multi-phase kernel: split into separate kernels per phase
                        section += "\n**IMPORTANT: Do NOT use `grid=(1,)`.** "
                        section += "This kernel updates **multiple arrays** in separate phases. "
                        section += "Split into **separate Triton kernels** launched sequentially from Python "
                        section += "(one per phase/statement). Within each kernel, parallelize the spatial "
                        section += "dimensions — kernel launch barriers between phases handle the dependencies.\n"
                        section += "\nExample pattern:\n"
                        section += "```python\n"
                        section += "for t in range(TSTEPS):  # timestep loop in Python\n"
                        section += "    phase1_kernel[grid](...)  # parallelize i (or i,j) within phase\n"
                        section += "    phase2_kernel[grid](...)  # next phase, different arrays\n"
                        section += "```\n"
                        section += "\n**Minimize kernel launches**: Fuse compatible phases into a single kernel "
                        section += "when they operate on independent data within each thread. For example, "
                        section += "a forward sweep and backward sweep on the same row can share a kernel.\n"
                    else:
                        # Single write array with self-deps: truly sequential
                        section += "\nThis kernel reads and writes the **same array** with neighbor dependencies. "
                        section += "Use `grid=(1,)` with nested loops to process elements sequentially.\n"
                    analysis_sections.append(section)

        # Cross-reference WAR scoping with parallelization options
        if (war_result and war_result.get('loop_level_scoping')
                and par_result and par_result.get('options')):
            loop_scoping = war_result['loop_level_scoping']
            copies = war_result.get('arrays_needing_copy', [])
            par_dims = par_result.get('dims', [])

            recommendations = []
            for opt in par_result.get('options', []):
                if not opt.get('valid'):
                    continue
                pdim = opt['parallel_dim']
                # Handle multi-dim parallel dims (e.g., "i, j, k" from N-D analysis)
                pdim_list = [d.strip() for d in pdim.split(',')]
                # Check if ALL WAR arrays are safe at ALL parallel dimensions
                all_safe = True
                needs_copy_arrs = []
                for arr in copies:
                    scoping = loop_scoping.get(arr, {})
                    safe = scoping.get('safe_to_parallelize_loops', [])
                    # All parallel dims must be safe for this array
                    if not all(d in safe for d in pdim_list):
                        all_safe = False
                        needs_copy_arrs.append(arr)

                if all_safe:
                    recommendations.append(
                        f"- **RECOMMENDED**: Parallelize `{pdim}` — no WAR copies needed for any array"
                    )
                elif needs_copy_arrs:
                    recommendations.append(
                        f"- Parallelize `{pdim}`: must clone {', '.join(f'`{a}`' for a in needs_copy_arrs)} before parallel region"
                    )

            if recommendations:
                section = "\n## WAR + Parallelization Recommendation\n\n"
                section += "\n".join(recommendations)
                section += "\n"
                analysis_sections.append(section)

        if scalar_exp_result and scalar_exp_result.get('has_scalar_expansion'):
            if HAS_SCALAR_EXPANSION and format_scalar_expansion_for_prompt:
                try:
                    formatted = format_scalar_expansion_for_prompt(kernel_name, scalar_exp_result)
                    if formatted:
                        analysis_sections.append(f"\n{formatted}\n")
                except Exception:
                    pass

            # Check if scalar expansion unblocks parallelization
            if par_result and par_result.get('options'):
                scalar_unblocked = []
                for opt in par_result['options']:
                    if not opt['valid']:
                        issues = opt.get('issues', [])
                        wc_issues = [iss for iss in issues if 'Write conflict' in iss]
                        # All write conflicts are on scalars (empty subscript "write []")
                        if wc_issues and all('write []' in iss for iss in wc_issues):
                            has_cross_phase = any('Cross-phase' in iss for iss in issues)
                            if not has_cross_phase:
                                scalar_unblocked.append(opt['parallel_dim'])
                if scalar_unblocked:
                    section = ("\n## Post-Expansion Parallelization\n\n"
                               f"After scalar expansion, `{'`, `'.join(scalar_unblocked)}` "
                               "becomes safely parallelizable.\n"
                               f"Use `grid=(N,)` to parallelize `{scalar_unblocked[0]}` "
                               "with the other dimension sequential inside the kernel.\n"
                               "Use **BLOCK_SIZE >= 64** for inner-loop vectorization.\n")
                    analysis_sections.append(section)

            # Fix 10: When scalar expansion is present AND cross-phase deps exist
            # AND no valid parallelization option, the phases should be split into
            # separate kernels with row/column-parallel execution.
            # (e.g., deriche: IIR recurrence along j, rows i are independent)
            if (not any(o['valid'] for o in par_result['options'])
                    and any('Cross-phase' in iss
                            for o in par_result['options']
                            for iss in o.get('issues', []))):
                section = "\n## Multi-Phase Row/Column-Parallel Execution\n\n"
                section += "**Each phase (top-level `for` loop) is independently parallelizable.**\n\n"
                section += "**CRITICAL: Minimize kernel launches by FUSING phases that share the same grid.**\n"
                section += "Fuse forward pass, backward pass, AND element-wise combination into a SINGLE kernel per direction.\n"
                section += "This gives **2 kernel launches** (horizontal + vertical), NOT 6.\n\n"
                section += "**For each fused direction kernel**:\n"
                section += "- The OUTER loop dimension becomes the grid: `grid = (OUTER_DIM,)`\n"
                section += "- One CTA per row (or column). Run forward IIR, backward IIR, and combination "
                section += "ALL within the same CTA — they operate on the same row/column and can share data.\n"
                section += "- Scalar variables (after expansion) become per-CTA registers — "
                section += "no array allocation needed.\n\n"
                section += "**Example** for a fused horizontal kernel (forward + backward + combine):\n"
                section += "```python\n"
                section += "@triton.jit\n"
                section += "def horiz_kernel(imgOut_ptr, imgIn_ptr, yy1_ptr, y2_ptr,\n"
                section += "                 a1, a2, a3, a4, b1, b2, c1, H: tl.constexpr, W: tl.constexpr):\n"
                section += "    row = tl.program_id(0)  # One CTA per row\n"
                section += "    # --- Forward pass ---\n"
                section += "    ym1 = 0.0; ym2 = 0.0; xm1 = 0.0\n"
                section += "    for j in range(H):\n"
                section += "        val = tl.load(imgIn_ptr + row * H + j)\n"
                section += "        out = a1 * val + a2 * xm1 + b1 * ym1 + b2 * ym2\n"
                section += "        tl.store(yy1_ptr + row * H + j, out)\n"
                section += "        xm1 = val; ym2 = ym1; ym1 = out\n"
                section += "    # --- Backward pass ---\n"
                section += "    yp1 = 0.0; yp2 = 0.0; xp1 = 0.0; xp2 = 0.0\n"
                section += "    for j in range(H-1, -1, -1):\n"
                section += "        val = tl.load(imgIn_ptr + row * H + j)\n"
                section += "        out = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2\n"
                section += "        tl.store(y2_ptr + row * H + j, out)\n"
                section += "        xp2 = xp1; xp1 = val; yp2 = yp1; yp1 = out\n"
                section += "    # --- Combine ---\n"
                section += "    for j in range(H):\n"
                section += "        idx = row * H + j\n"
                section += "        imgOut = c1 * (tl.load(yy1_ptr + idx) + tl.load(y2_ptr + idx))\n"
                section += "        tl.store(imgOut_ptr + idx, imgOut)\n"
                section += "grid = (W,)  # W rows processed in parallel\n"
                section += "```\n\n"
                section += "Similarly fuse vertical forward + backward + combine into a SINGLE kernel with `grid = (H,)`.\n\n"
                section += "**IMPORTANT**: Do NOT use `grid=(1,)`. Each row/column is independent — "
                section += "use multi-CTA parallelism.\n\n"
                section += "**Coefficient computation**: Compute ALL coefficients (a1, a2, b1, b2, etc.) "
                section += "in the **Python wrapper** using `import math; math.exp(...)`, NOT `torch.exp()` "
                section += "or `tl.exp()`. Pass them as plain Python floats to the kernel.\n"
                section += "```python\n"
                section += "import math\n"
                section += "def wrapper(imgIn, imgOut, ..., alpha, H, W):\n"
                section += "    k = (1-math.exp(-alpha))**2 / (1+2*alpha*math.exp(-alpha)-math.exp(2*alpha))\n"
                section += "    a1 = k; a2 = k*math.exp(-alpha)*(alpha-1); ...\n"
                section += "    b1 = 2.0**(-alpha); b2 = -math.exp(-2*alpha)\n"
                section += "    horiz_kernel[(W,)](imgOut, imgIn, yy1, y2, a1, a2, a3, a4, b1, b2, c1, H, W)\n"
                section += "    vert_kernel[(H,)](imgOut, yy1, y2, a5, a6, a7, a8, b1, b2, c2, H, W)\n"
                section += "```\n"
                analysis_sections.append(section)

        if reduction_result and reduction_result.get('is_reduction'):
            if HAS_REDUCTION and build_reduction_instructions:
                try:
                    formatted = build_reduction_instructions(reduction_result)
                    if formatted:
                        analysis_sections.append(f"\n{formatted}\n")
                except Exception:
                    pass

        # GPU parallelization strategy (wavefront, inner-loop vectorization, multi-GEMM)
        if HAS_GPU_STRATEGY:
            try:
                kernel_file = os.path.join(POLYBENCH_KERNELS_DIR, f"{kernel_name}.c")
                gpu_strategy = analyze_kernel_gpu_strategy(kernel_name, kernel_file)
                if gpu_strategy:
                    # Skip inner_loop_vectorization — it forces grid=(1,) which
                    # prevents the LLM from discovering better parallelism
                    # (e.g., ludcmp: LLM can parallelize j without this constraint)
                    pattern = gpu_strategy.get('pattern', '')
                    if pattern == 'inner_loop_vectorization':
                        pass  # Skip — grid=(1,) is too conservative
                    else:
                        formatted = build_gpu_strategy_instructions(kernel_name, gpu_strategy)
                        if formatted:
                            analysis_sections.append(f"\n{formatted}\n")
            except Exception:
                pass

        # Fix 6: General anti-patterns guidance (always appended when analysis is enabled)
        anti_patterns = """
## Common Mistakes to AVOID

1. **NEVER use Python for-loops to launch many small kernels.** If you have sequential dims `r, q` and parallel dim `p`, do NOT write `for r in range(NR): for q in range(NQ): kernel[grid](...)`. Instead, fuse them: `grid = (NR * NQ * cdiv(NP, BLOCK),)` and inside the kernel compute `r = pid // (NQ * NP_blocks)`, `q = (pid // NP_blocks) % NQ`, `p_start = (pid % NP_blocks) * BLOCK`.

2. **NEVER write parallel kernels then call a serial fallback.** If you write tiled/vectorized phase kernels (e.g., `phase1_kernel`, `phase2_kernel`), the host function MUST call them. Do NOT add a serial `grid=(1,)` wrapper that ignores the parallel implementations.

3. **Use `tl.dot()` for ALL matrix multiply and outer-product accumulations.** NEVER use scalar triple-nested loops (`for i: for j: for k: acc += a[i,k]*b[k,j]`) inside a Triton kernel. Tile with BLOCK_SIZE and use `acc += tl.dot(a_tile, b_tile)`.

4. **For stencil kernels with `grid=(1,)` guidance: NEVER use multi-block grids.** If the analysis says `grid=(1,)`, do NOT use `grid=(triton.cdiv(N, BLOCK),)`. Multiple CTAs cannot synchronize between phases — CTA 0 may read stale values written by CTA 1. Use `BLOCK = triton.next_power_of_2(N)` so one CTA covers ALL elements.

5. **Cap BLOCK_SIZE at 128 for vectorized dimensions** (except `grid=(1,)` kernels). Use `BLOCK = min(triton.next_power_of_2(N), 128)` and iterate in chunks with `for j_start in range(0, N, BLOCK)`. Large BLOCK sizes (256+) waste registers and kill GPU occupancy. Use BLOCK_SIZE = 64 for inner reduction loops.

6. **Minimize kernel launches.** Fuse phases that share the same grid dimension into a SINGLE kernel (e.g., forward pass + backward pass + combine on the same row). Each kernel launch adds ~5-10μs overhead that dominates at small problem sizes.
"""
        analysis_sections.append(anti_patterns)

    analysis_text = "\n".join(analysis_sections)

    prompt = f"""I have a Polybench/C kernel that I want to implement in Triton for GPU acceleration.

## Original C Code:
```c
{source}
```

## Kernel Loop to Implement:
```c
{loop_code}
```
{analysis_text}

## Array Information:
{array_info}

## Dimension Parameters (compile-time constants in C, runtime parameters in Triton):
{dim_info}

## Requirements:
Please generate a complete Triton implementation that:
1. Includes a @triton.jit kernel function named `{func_id}_kernel`
2. Includes a Python wrapper function named `{func_id}_triton`
3. The wrapper accepts tensor arrays, scalar parameters, and dimension parameters
4. Uses appropriate block sizes and memory access patterns
5. Handles edge cases with masking
6. Is functionally equivalent to the C code (same computation, same results)
7. For 2D arrays, compute linear index as `row * stride + col`
8. For 3D arrays, compute linear index as `dim0 * (dim1_size * dim2_size) + dim1 * dim2_size + dim2`

## REQUIRED function signature (use EXACTLY these parameter names):
```python
def {func_id}_triton({exact_sig}):
    ...  # kernel computation
```

## CRITICAL: Triton Compilation Rules

**Pass dimension parameters as `tl.constexpr`** for best performance:
```python
# GOOD — enables compile-time unrolling and constant folding
def kernel(ptr, N: tl.constexpr, M: tl.constexpr, BLOCK: tl.constexpr):
    for i in range(N):  # compiler can unroll this
        ...
```

**NEVER use `tl.arange()` inside a for loop:**
```python
# WRONG
for block_start in range(0, n, BLOCK_SIZE):
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # ERROR!

# CORRECT
offsets = tl.arange(0, BLOCK_SIZE)  # Define once at start
for block_start in range(0, n, BLOCK_SIZE):
    current_offsets = block_start + offsets
```

**Prefer vectorized access over scalar indexing inside @triton.jit kernel:**
```python
# WRONG — scalar indexing into a Triton tensor
for i in range(BLOCK_SIZE):
    val = tensor[i]

# CORRECT — vectorized load
mask = offsets < n_elements
vals = tl.load(ptr + offsets, mask=mask)
```
**Exception**: For variable-length inner reductions (e.g., neighbor traversal),
scalar `for` loops with `tl.load`/`tl.store` on pointers are correct.
Store results directly with `tl.store(ptr + idx, val)` — do NOT use
`tl.where(arange == idx, val, vec)` which is O(BLOCK_SIZE) per element.

**NEVER use non-existent Triton functions:**
- Use Python operators: `a * b`, `a / b`, `a + b` (not `tl.mul`, `tl.div`, `tl.add`)
- Use `triton.cdiv()` in wrapper only

**NEVER use Python lists, break/continue inside @triton.jit kernels**
**Pass tensors directly to kernels, NOT data_ptr()**
**NEVER use chained comparisons (use separate comparisons with &)**

Provide ONLY the Python code, no additional explanation."""

    return prompt
