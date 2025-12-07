import torch
import triton
import triton.language as tl

@triton.jit
def s276_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, mid, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    
    # Condition: i+1 < mid
    condition = (offsets + 1) < mid
    
    # Compute based on condition
    result1 = a_vals + b_vals * c_vals  # for i+1 < mid
    result2 = a_vals + b_vals * d_vals  # for i+1 >= mid
    
    # Select based on condition
    final_result = tl.where(condition, result1, result2)
    
    # Store result
    tl.store(a_ptr + offsets, final_result, mask=mask)

def s276_triton(a, b, c, d, mid):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s276_kernel[grid](
        a, b, c, d,
        n_elements, mid,
        BLOCK_SIZE=BLOCK_SIZE
    )