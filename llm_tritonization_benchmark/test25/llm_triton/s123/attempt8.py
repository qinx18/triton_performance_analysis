import triton
import triton.language as tl
import torch

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, a_size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    mask = (block_start + offsets) < n_elements
    
    # Load input arrays
    b_vals = tl.load(b_ptr + block_start + offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + block_start + offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + block_start + offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + block_start + offsets, mask=mask, other=0.0)
    
    # Compute values
    base_vals = b_vals + d_vals * e_vals
    cond_vals = c_vals + d_vals * e_vals
    
    # Store at j = 2*i
    j_base = 2 * (block_start + offsets)
    j_base_mask = mask & (j_base < a_size)
    tl.store(a_ptr + j_base, base_vals, mask=j_base_mask)
    
    # Store at j = 2*i + 1 when condition is met
    cond_mask = c_vals > 0.0
    j_cond = j_base + 1
    j_cond_mask = mask & cond_mask & (j_cond < a_size)
    tl.store(a_ptr + j_cond, cond_vals, mask=j_cond_mask)

def s123_triton(a, b, c, d, e):
    n_elements = a.shape[0] // 2
    a_size = a.shape[0]
    BLOCK_SIZE = 256
    
    # Clear array a
    a.zero_()
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s123_kernel[grid](
        a, b, c, d, e,
        n_elements,
        a_size,
        BLOCK_SIZE=BLOCK_SIZE
    )