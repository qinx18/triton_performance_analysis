import torch
import triton
import triton.language as tl

@triton.jit
def s161_phase1_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Phase 1: When b[i] < 0, write c[i+1] = a[i] + d[i] * d[i]
    This must happen FIRST before phase 2 reads c values.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    # Load data
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c_next_vals = tl.load(c_ptr + offsets + 1, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + offsets, mask=mask, other=0.0)

    # Condition: b[i] < 0
    cond_mask = b_vals < 0.0

    # When b[i] < 0: c[i+1] = a[i] + d[i] * d[i]
    c_next_new = a_vals + d_vals * d_vals
    c_next_result = tl.where(cond_mask, c_next_new, c_next_vals)

    # Store updated c[i+1]
    tl.store(c_ptr + offsets + 1, c_next_result, mask=mask)


@triton.jit
def s161_phase2_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Phase 2: When b[i] >= 0, write a[i] = c[i] + d[i] * e[i]
    This reads c[i] which may have been updated in phase 1.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    # Load data
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + offsets, mask=mask, other=0.0)

    # Condition: b[i] >= 0
    cond_mask = b_vals >= 0.0

    # When b[i] >= 0: a[i] = c[i] + d[i] * e[i]
    a_new = c_vals + d_vals * e_vals
    a_result = tl.where(cond_mask, a_new, a_vals)

    # Store updated a[i]
    tl.store(a_ptr + offsets, a_result, mask=mask)


def s161_triton(a, b, c, d, e):
    """
    Corrected Triton implementation of TSVC s161 - following true C sequential semantics.

    Uses two-phase approach:
    1. Phase 1: Process b[i] < 0 (write c[i+1])
    2. Phase 2: Process b[i] >= 0 (read c[i], write a[i])

    This ensures RAW dependencies are properly handled.
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    e = e.contiguous()

    n = len(a) - 1

    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)

    # Phase 1: Handle b[i] < 0 case (write c[i+1])
    s161_phase1_kernel[grid](
        a, b, c, d,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Phase 2: Handle b[i] >= 0 case (read c[i], write a[i])
    s161_phase2_kernel[grid](
        a, b, c, d, e,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return a, c
