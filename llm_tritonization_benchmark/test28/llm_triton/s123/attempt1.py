import triton
import triton.language as tl
import torch

@triton.jit
def s123_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_half,
    BLOCK_SIZE: tl.constexpr,
):
    # This kernel processes each element sequentially since j depends on conditional increments
    pid = tl.program_id(0)
    
    if pid >= n_half:
        return
    
    i = pid
    
    # Calculate j based on how many positive c values we've seen up to position i
    j = i  # Base increment for each iteration
    
    # Count additional increments from positive c values in range [0, i)
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process in blocks to count positive c values
    additional_j = 0
    for block_start in range(0, i, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < i
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        pos_mask = c_vals > 0.0
        additional_j += tl.sum(tl.where(pos_mask, 1, 0))
    
    j = j + additional_j
    
    # Load values for current i
    b_val = tl.load(b_ptr + i)
    c_val = tl.load(c_ptr + i)
    d_val = tl.load(d_ptr + i)
    e_val = tl.load(e_ptr + i)
    
    # First assignment: a[j] = b[i] + d[i] * e[i]
    result1 = b_val + d_val * e_val
    tl.store(a_ptr + j, result1)
    
    # Conditional assignment
    if c_val > 0.0:
        j = j + 1
        result2 = c_val + d_val * e_val
        tl.store(a_ptr + j, result2)

def s123_triton(a, b, c, d, e):
    n = b.shape[0]
    n_half = n // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_half, BLOCK_SIZE),)
    
    s123_kernel[grid](
        a, b, c, d, e,
        n_half,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a