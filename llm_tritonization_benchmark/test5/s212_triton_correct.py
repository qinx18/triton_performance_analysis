import torch
import triton
import triton.language as tl

@triton.jit
def s212_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, a_orig_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Corrected Triton kernel for s212: a[i] *= c[i]; b[i] += a[i + 1] * d[i]
    Uses a_orig to avoid race condition when reading a[i+1]
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load data with masking for bounds checking
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + offsets, mask=mask, other=0.0)

    # First operation: a[i] *= c[i]
    a_updated = a_vals * c_vals
    tl.store(a_ptr + offsets, a_updated, mask=mask)

    # Load a_orig[i+1] for second operation (ORIGINAL values, not modified)
    offsets_plus_1 = offsets + 1
    mask_plus_1 = mask & (offsets_plus_1 < (n_elements + 1))
    a_orig_plus_1 = tl.load(a_orig_ptr + offsets_plus_1, mask=mask_plus_1, other=0.0)

    # Second operation: b[i] += a_orig[i + 1] * d[i]
    b_updated = b_vals + a_orig_plus_1 * d_vals
    tl.store(b_ptr + offsets, b_updated, mask=mask)

def s212_triton(a, b, c, d):
    """
    Corrected Triton implementation of TSVC s212

    Saves original `a` values to avoid race condition, just like s211 does with `b`.
    """
    # Ensure contiguous memory layout for optimal GPU memory access
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()

    # Save original a values before modification (FIX!)
    a_orig = a.clone()

    # Process n-1 elements as per original algorithm
    n_elements = a.shape[0] - 1

    if n_elements <= 0:
        return a, b

    # Use power-of-2 block size for optimal memory coalescing
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Launch kernel with a_orig to avoid race condition
    s212_kernel[grid](
        a, b, c, d, a_orig,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return a, b
