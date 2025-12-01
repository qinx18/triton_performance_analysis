import torch
import triton
import triton.language as tl

@triton.jit
def s124_kernel(b_ptr, c_ptr, d_ptr, e_ptr, a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process elements sequentially in blocks
    for start_idx in range(0, n_elements, BLOCK_SIZE):
        current_offsets = start_idx + offsets
        mask = current_offsets < n_elements
        
        # Load input arrays
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
        e_vals = tl.load(e_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute conditionally
        de_product = d_vals * e_vals
        condition = b_vals > 0.0
        result = tl.where(condition, b_vals + de_product, c_vals + de_product)
        
        # Store results (j = i since j increments for every i)
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s124_triton(a, b, c, d, e):
    n_elements = b.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s124_kernel[grid](
        b, c, d, e, a, n_elements, BLOCK_SIZE
    )