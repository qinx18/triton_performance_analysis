import triton
import triton.language as tl
import torch

@triton.jit
def s276_kernel(a_ptr, b_ptr, c_ptr, d_ptr, mid, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load arrays
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    
    # Condition: i+1 < mid
    condition = (idx + 1) < mid
    combined_mask = mask & condition
    
    # First branch: a[i] += b[i] * c[i]
    result1 = a_vals + b_vals * c_vals
    
    # Second branch: a[i] += b[i] * d[i]  
    result2 = a_vals + b_vals * d_vals
    
    # Select based on condition
    final_result = tl.where(condition, result1, result2)
    
    # Store result
    tl.store(a_ptr + idx, final_result, mask=mask)

def s276_triton(a, b, c, d, mid):
    N = a.shape[0]
    BLOCK_SIZE = 1024
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s276_kernel[grid](
        a, b, c, d,
        mid, N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a