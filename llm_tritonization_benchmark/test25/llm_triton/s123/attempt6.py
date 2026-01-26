import triton
import triton.language as tl
import torch

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    mask = (block_start + offsets) < n_elements
    
    # Load input arrays
    b_vals = tl.load(b_ptr + block_start + offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + block_start + offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + block_start + offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + block_start + offsets, mask=mask, other=0.0)
    
    # Compute base values
    base_vals = b_vals + d_vals * e_vals
    
    # Compute conditional values
    cond_vals = c_vals + d_vals * e_vals
    cond_mask = c_vals > 0.0
    
    # Store base values at 2*i positions
    a_base_offsets = 2 * (block_start + offsets)
    a_base_mask = mask & (a_base_offsets < 2 * n_elements)
    tl.store(a_ptr + a_base_offsets, base_vals, mask=a_base_mask)
    
    # Store conditional values at 2*i+1 positions when condition is true
    a_cond_offsets = a_base_offsets + 1
    a_cond_mask = mask & cond_mask & (a_cond_offsets < 2 * n_elements)
    tl.store(a_ptr + a_cond_offsets, cond_vals, mask=a_cond_mask)

def s123_triton(a, b, c, d, e):
    n_elements = b.shape[0] // 2
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s123_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )