import torch
import triton
import triton.language as tl

@triton.jit
def s124_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel processes the entire array sequentially since j depends on previous iterations
    # We need to handle the sequential nature of the induction variable j
    
    # Load all elements for this block
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input arrays
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + offsets, mask=mask, other=0.0)
    
    # Compute the result based on condition
    condition = b_vals > 0.0
    result = tl.where(condition, b_vals + d_vals * e_vals, c_vals + d_vals * e_vals)
    
    # Store results - the output index is the same as input index since j increments every iteration
    tl.store(a_ptr + offsets, result, mask=mask)

def s124_triton(a, b, c, d, e):
    n_elements = b.numel()
    
    # Since j increments in every iteration regardless of condition,
    # the mapping is simply i -> j where j = i (starting from j = 0)
    # This means we can parallelize this computation
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s124_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a