import triton
import triton.language as tl
import torch

@triton.jit
def s272_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, t, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load values
    e_vals = tl.load(e_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Conditional computation
    condition = e_vals >= t
    
    # Compute updates
    cd_product = c_vals * d_vals
    c_squared = c_vals * c_vals
    
    # Apply conditional updates
    a_vals = tl.where(condition, a_vals + cd_product, a_vals)
    b_vals = tl.where(condition, b_vals + c_squared, b_vals)
    
    # Store results
    tl.store(a_ptr + indices, a_vals, mask=mask)
    tl.store(b_ptr + indices, b_vals, mask=mask)

def s272_triton(a, b, c, d, e, t):
    N = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s272_kernel[grid](a, b, c, d, e, t, N, BLOCK_SIZE=BLOCK_SIZE)