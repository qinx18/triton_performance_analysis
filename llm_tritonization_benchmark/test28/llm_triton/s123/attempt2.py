import triton
import triton.language as tl
import torch

@triton.jit
def s123_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_half,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    if pid >= n_half:
        return
    
    i = pid
    
    # Calculate j by counting how many elements we've processed
    # j starts at 0 (since we increment before first assignment in C)
    # For each element up to i, we get +1, plus +1 more if c[k] > 0
    j = i  # Base increment for reaching position i
    
    # Count additional increments from positive c values in range [0, i)
    additional_j = 0
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, i, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < i
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        pos_count = tl.sum(tl.where(c_vals > 0.0, 1, 0))
        additional_j += pos_count
    
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
    grid = (n_half,)
    
    s123_kernel[grid](
        a, b, c, d, e,
        n_half,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a