import torch
import triton
import triton.language as tl

@triton.jit
def s2244_kernel(
    a_ptr, b_ptr, c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s2244 - vectorizable after removing false dependency.

    Original pattern appears to have dependency:
        a[i+1] = b[i] + e[i]
        a[i] = b[i] + c[i]

    But a[i+1] from iteration i is overwritten by a[i] in iteration i+1,
    except for the last element. So we can vectorize a[i] = b[i] + c[i]
    for all i in [0, n_elements-2].
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements - 1

    # Load values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)

    # Compute a[i] = b[i] + c[i] (vectorized)
    a_vals = b_vals + c_vals

    # Store results
    tl.store(a_ptr + offsets, a_vals, mask=mask)

@triton.jit
def s2244_last_element_kernel(a_ptr, b_ptr, e_ptr, last_idx):
    """
    Handle the last element which is not overwritten:
    a[n_elements-1] = b[n_elements-2] + e[n_elements-2]
    """
    if tl.program_id(0) == 0:
        # Load values
        b_last_minus_1 = tl.load(b_ptr + last_idx - 1)
        e_last_minus_1 = tl.load(e_ptr + last_idx - 1)

        # Compute
        a_last = b_last_minus_1 + e_last_minus_1

        # Store
        tl.store(a_ptr + last_idx, a_last)

def s2244_triton(a, b, c, e):
    """
    Triton implementation of TSVC s2244 - optimized by recognizing false dependency.

    The original code appears to have dependencies:
        a[i+1] = b[i] + e[i]
        a[i] = b[i] + c[i]

    Analysis shows that a[i+1] written in iteration i is overwritten by a[i]
    in iteration i+1, except for the last element. This allows vectorization.
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    e = e.contiguous()

    len_1d = a.size(0)

    if len_1d <= 1:
        return a

    # Vectorized processing for elements 0 to len_1d-2
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(len_1d - 1, BLOCK_SIZE)

    s2244_kernel[(grid_size,)](
        a, b, c,
        len_1d,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Handle the last element separately
    # a[len_1d-1] = b[len_1d-2] + e[len_1d-2]
    s2244_last_element_kernel[(1,)](
        a, b, e, len_1d - 1
    )

    return a
