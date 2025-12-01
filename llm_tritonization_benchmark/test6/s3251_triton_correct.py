import torch
import triton
import triton.language as tl

@triton.jit
def s3251_expand_array_kernel(
    a_expanded_ptr, a_orig_ptr, b_ptr, c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel 1: Array expansion - compute a_expanded where:
    a_expanded[0] = a_orig[0]  (original value)
    a_expanded[i] = b[i-1] + c[i-1] for i > 0

    This expands a into an array with the values needed by each iteration.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # For i=0, a_expanded[0] = a_orig[0]
    # For i>0, a_expanded[i] = b[i-1] + c[i-1]

    # Load b[i-1] and c[i-1] values (shifted by 1)
    offsets_minus_1 = offsets - 1
    mask_minus_1 = (offsets > 0) & (offsets < n_elements)

    b_prev = tl.load(b_ptr + offsets_minus_1, mask=mask_minus_1, other=0.0)
    c_prev = tl.load(c_ptr + offsets_minus_1, mask=mask_minus_1, other=0.0)

    # Compute a_expanded values
    a_vals = b_prev + c_prev

    # For offset 0, override with original a[0]
    a_orig_0 = tl.load(a_orig_ptr + offsets, mask=(offsets == 0), other=0.0)
    a_vals = tl.where(offsets == 0, a_orig_0, a_vals)

    # Store a_expanded
    tl.store(a_expanded_ptr + offsets, a_vals, mask=mask)


@triton.jit
def s3251_compute_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, a_expanded_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel 2: Vectorized computation using expanded array
    a[i+1] = b[i] + c[i]
    b[i] = c[i] * e[i]
    d[i] = a_expanded[i] * e[i]
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (n_elements - 1)

    # Load values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    a_expanded_vals = tl.load(a_expanded_ptr + offsets, mask=mask)

    # Compute a[i+1] = b[i] + c[i]
    a_new = b_vals + c_vals
    tl.store(a_ptr + offsets + 1, a_new, mask=mask)

    # Compute b[i] = c[i] * e[i]
    b_new = c_vals * e_vals
    tl.store(b_ptr + offsets, b_new, mask=mask)

    # Compute d[i] = a_expanded[i] * e[i]
    d_new = a_expanded_vals * e_vals
    tl.store(d_ptr + offsets, d_new, mask=mask)


def s3251_triton(a, b, c, d, e):
    """
    Triton implementation of TSVC s3251 using array expansion technique.

    Original pattern has dependency:
        for i in range(N-1):
            a[i+1] = b[i] + c[i]   # Writes to a[i+1]
            b[i] = c[i] * e[i]
            d[i] = a[i] * e[i]     # Reads a[i] (written in iteration i-1)

    Array expansion breaks the dependency:
    1. Expand a into array a_expanded where:
       - a_expanded[0] = a_orig[0] (original value)
       - a_expanded[i] = b[i-1] + c[i-1] for i > 0

    2. Vectorize the computation:
       - a[i+1] = b[i] + c[i]
       - b[i] = c[i] * e[i]
       - d[i] = a_expanded[i] * e[i]

    This allows full parallelization!
    """
    a = a.contiguous().clone()
    b = b.contiguous().clone()
    d = d.contiguous().clone()

    LEN_1D = len(a)
    n_elements = LEN_1D

    # Create temporary array for array expansion
    a_expanded = torch.empty_like(a)

    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Step 1: Expand array a into a_expanded
    s3251_expand_array_kernel[grid](
        a_expanded, a, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Step 2: Vectorized computation using a_expanded
    s3251_compute_kernel[grid](
        a, b, c, d, e, a_expanded,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return a, b, d
