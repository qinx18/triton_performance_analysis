import torch
import triton
import triton.language as tl

@triton.jit
def s1244_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    mask = current_offsets < n_elements
    
    # Load data
    b_vals = tl.load(b_ptr + current_offsets, mask=mask)
    c_vals = tl.load(c_ptr + current_offsets, mask=mask)
    
    # Compute a[i] = b[i] + c[i] * c[i] + b[i]*b[i] + c[i]
    c_squared = c_vals * c_vals
    b_squared = b_vals * b_vals
    a_vals = b_vals + c_squared + b_squared + c_vals
    
    # Store a[i] values
    tl.store(a_ptr + current_offsets, a_vals, mask=mask)
    
    # Load a[i+1] from copy for d[i] computation
    next_offsets = current_offsets + 1
    next_mask = next_offsets < (n_elements + 1)
    a_next_vals = tl.load(a_copy_ptr + next_offsets, mask=next_mask)
    
    # Compute d[i] = a[i] + a[i+1]
    d_vals = a_vals + a_next_vals
    
    # Store d[i] values
    tl.store(d_ptr + current_offsets, d_vals, mask=mask)

def s1244_triton(a, b, c, d):
    n_elements = a.shape[0] - 1  # Loop goes to LEN_1D-1
    
    # Create read-only copy for WAR dependency handling
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1244_kernel[grid](
        a, a_copy, b, c, d, n_elements, BLOCK_SIZE
    )