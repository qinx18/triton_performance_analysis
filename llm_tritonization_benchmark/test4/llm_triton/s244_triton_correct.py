import torch
import triton
import triton.language as tl

@triton.jit
def s244_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s244 - vectorizable after removing false dependency.

    Original pattern appears to have dependency:
        a[i] = b[i] + c[i] * d[i]
        b[i] = c[i] + b[i]
        a[i+1] = b[i] + a[i+1] * d[i]

    But a[i+1] is overwritten in the next iteration (except the last element),
    so we can vectorize the first two statements for all i < n_elements-1.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements - 1

    # Load values
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)

    # Compute updates (vectorized)
    new_a = b_vals + c_vals * d_vals
    new_b = c_vals + b_vals

    # Store results
    tl.store(a_ptr + offsets, new_a, mask=mask)
    tl.store(b_ptr + offsets, new_b, mask=mask)

@triton.jit
def s244_last_element_kernel(a_ptr, b_ptr, d_ptr, last_idx):
    """
    Handle the last element which is not overwritten:
    a[LEN_1D-1] = b[LEN_1D-2] + a[LEN_1D-1] * d[LEN_1D-2]
    """
    if tl.program_id(0) == 0:
        # Load values
        b_last_minus_1 = tl.load(b_ptr + last_idx - 1)
        a_last = tl.load(a_ptr + last_idx)
        d_last_minus_1 = tl.load(d_ptr + last_idx - 1)

        # Compute
        new_a_last = b_last_minus_1 + a_last * d_last_minus_1

        # Store
        tl.store(a_ptr + last_idx, new_a_last)

def s244_triton(a, b, c, d):
    """
    Triton implementation of TSVC s244 - optimized by recognizing false dependency.

    The original code appears to have dependencies, but analysis shows that
    a[i+1] written in iteration i is always overwritten by a[i+1] in iteration i+1,
    except for the last element. This allows vectorization.
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()

    LEN_1D = a.shape[0]

    if LEN_1D <= 1:
        return a, b

    # Vectorized processing for elements 0 to LEN_1D-2
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(LEN_1D - 1, BLOCK_SIZE)

    s244_kernel[(grid_size,)](
        a, b, c, d,
        LEN_1D,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Handle the last element separately
    # a[LEN_1D-1] = b[LEN_1D-2] + a[LEN_1D-1] * d[LEN_1D-2]
    s244_last_element_kernel[(1,)](
        a, b, d, LEN_1D - 1
    )

    return a, b
