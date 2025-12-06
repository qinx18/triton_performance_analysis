import torch
import triton
import triton.language as tl

@triton.jit
def s211_kernel(a_ptr, b_ptr, b_copy_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets + 1  # Start from index 1
    
    mask = (indices >= 1) & (indices < n_elements - 1)
    
    # Load data with appropriate offsets
    b_prev = tl.load(b_copy_ptr + indices - 1, mask=mask)
    b_next = tl.load(b_copy_ptr + indices + 1, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    e_vals = tl.load(e_ptr + indices, mask=mask)
    
    # Compute results
    a_result = b_prev + c_vals * d_vals
    b_result = b_next - e_vals * d_vals
    
    # Store results
    tl.store(a_ptr + indices, a_result, mask=mask)
    tl.store(b_ptr + indices, b_result, mask=mask)

def s211_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create read-only copy of b to handle WAR dependency
    b_copy = b.clone()
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s211_kernel[grid](
        a, b, b_copy, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )