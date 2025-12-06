import torch
import triton
import triton.language as tl

@triton.jit
def s124_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel processes the sequential logic by having each block handle a portion
    # and using atomic operations to maintain the sequential nature of j increment
    
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load input data
    b_vals = tl.load(b_ptr + block_start + offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + block_start + offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + block_start + offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + block_start + offsets, mask=mask, other=0.0)
    
    # Compute d[i] * e[i] for all elements
    de_product = d_vals * e_vals
    
    # Compute results for both branches
    positive_result = b_vals + de_product
    negative_result = c_vals + de_product
    
    # Select based on condition b[i] > 0
    condition = b_vals > 0.0
    result = tl.where(condition, positive_result, negative_result)
    
    # Store results directly (j increments sequentially for each i)
    tl.store(a_ptr + block_start + offsets, result, mask=mask)

def s124_triton(a, b, c, d, e):
    n_elements = b.numel()
    
    # Since j increments for every i, the output array a uses the same indexing as input arrays
    # The key insight is that j starts at -1 and increments before assignment, so j maps 1:1 with i
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s124_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a