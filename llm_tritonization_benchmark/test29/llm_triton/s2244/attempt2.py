import triton
import triton.language as tl
import torch

@triton.jit
def s2244_kernel(
    a_ptr, b_ptr, c_ptr, e_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    # Get block offsets
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    # Mask for valid indices (i < n-1)
    mask = idx < (n - 1)
    
    # Load data for valid indices
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    e_vals = tl.load(e_ptr + idx, mask=mask)
    
    # Compute a[i] = b[i] + c[i] (this overwrites a[i+1] from previous iteration)
    a_vals = b_vals + c_vals
    tl.store(a_ptr + idx, a_vals, mask=mask)
    
    # For the last iteration only (i = n-2), compute a[i+1] = b[i] + e[i]
    # This handles the case where a[n-1] gets its final value
    last_iter_mask = mask & (idx == (n - 2))
    last_sum = tl.sum(last_iter_mask.to(tl.int32))
    if last_sum > 0:
        last_a_val = b_vals + e_vals
        tl.store(a_ptr + idx + 1, last_a_val, mask=last_iter_mask)

def s2244_triton(a, b, c, e):
    n = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s2244_kernel[grid](
        a, b, c, e,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a