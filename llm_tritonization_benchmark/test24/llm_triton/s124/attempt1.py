import triton
import triton.language as tl
import torch

@triton.jit
def s124_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel processes the sequential dependency using atomic operations
    # Each thread processes one element but writes sequentially
    
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + offsets, mask=mask, other=0.0)
    
    # Compute values based on condition
    condition = b_vals > 0.0
    result_vals = tl.where(condition, b_vals + d_vals * e_vals, c_vals + d_vals * e_vals)
    
    # Store results - each element maps to its original index since j increments for every i
    tl.store(a_ptr + offsets, result_vals, mask=mask)

def s124_triton(a, b, c, d, e):
    n_elements = b.shape[0]
    
    # Clear output array
    a.zero_()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s124_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )