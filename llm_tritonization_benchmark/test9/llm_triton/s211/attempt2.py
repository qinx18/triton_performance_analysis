import triton
import triton.language as tl
import torch

@triton.jit
def s211_kernel(a_ptr, b_ptr, b_copy_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets + 1
    
    mask = (indices >= 1) & (indices < n_elements - 1)
    
    # Load b[i-1] from read-only copy
    b_prev_indices = indices - 1
    b_prev_mask = mask & (b_prev_indices >= 0)
    b_prev = tl.load(b_copy_ptr + b_prev_indices, mask=b_prev_mask, other=0.0)
    
    # Load c[i] and d[i]
    c_vals = tl.load(c_ptr + indices, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + indices, mask=mask, other=0.0)
    
    # Load b[i+1] from read-only copy
    b_next_indices = indices + 1
    b_next_mask = mask & (b_next_indices < n_elements)
    b_next = tl.load(b_copy_ptr + b_next_indices, mask=b_next_mask, other=0.0)
    
    # Load e[i]
    e_vals = tl.load(e_ptr + indices, mask=mask, other=0.0)
    
    # Compute a[i] = b[i - 1] + c[i] * d[i]
    a_vals = b_prev + c_vals * d_vals
    
    # Compute b[i] = b[i + 1] - e[i] * d[i]
    b_vals = b_next - e_vals * d_vals
    
    # Store results
    tl.store(a_ptr + indices, a_vals, mask=mask)
    tl.store(b_ptr + indices, b_vals, mask=mask)

def s211_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    
    # Create read-only copy of b for race condition handling
    b_copy = b.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements - 2, BLOCK_SIZE),)
    
    s211_kernel[grid](
        a, b, b_copy, c, d, e, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )