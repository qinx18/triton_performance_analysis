import torch
import triton
import triton.language as tl

@triton.jit
def s221_update_a_kernel(
    a_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Parallel kernel to update array a.
    a[i] += c[i] * d[i] for i in [1, n_elements)
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE) + 1  # Start from index 1

    mask = offsets < n_elements

    # Load values
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + offsets, mask=mask, other=0.0)

    # Compute a[i] += c[i] * d[i]
    a_new = a_vals + c_vals * d_vals

    # Store result
    tl.store(a_ptr + offsets, a_new, mask=mask)

def s221_triton(a, b, c, d):
    """
    Optimized Triton implementation of TSVC s221

    Strategy:
    1. Parallelize a[i] updates using Triton kernel
    2. Use PyTorch cumsum for b[i] computation (prefix sum)

    This hybrid approach is much faster than sequential in-kernel loop:
    - a updates: Fully parallel on GPU
    - b computation: Efficient cumsum implementation

    Note: Triton doesn't have efficient built-in scan operations,
    so we use PyTorch's highly optimized cumsum for the prefix sum.
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()

    n = a.shape[0]

    if n < 2:
        return a, b

    # Step 1: Update all a[i] in parallel using Triton kernel
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n - 1, BLOCK_SIZE),)

    s221_update_a_kernel[grid](
        a, c, d,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Step 2: Compute b[i] using cumulative sum
    # b[i] = b[i-1] + a[i] + d[i]
    # b[1:] = b[0] + cumsum(a[1:] + d[1:])
    increments = a[1:] + d[1:]
    b[1:] = b[0] + torch.cumsum(increments, dim=0)

    return a, b
