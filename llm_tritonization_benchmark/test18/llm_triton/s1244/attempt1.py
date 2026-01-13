import triton
import triton.language as tl
import torch

@triton.jit
def s1244_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n_elements
    
    # Load b[i] and c[i]
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    
    # Compute a[i] = b[i] + c[i] * c[i] + b[i]*b[i] + c[i]
    a_vals = b_vals + c_vals * c_vals + b_vals * b_vals + c_vals
    
    # Store a[i]
    tl.store(a_ptr + idx, a_vals, mask=mask)
    
    # Load a[i+1] from the read-only copy
    idx_plus_1 = idx + 1
    mask_plus_1 = idx_plus_1 < (n_elements + 1)
    a_plus_1_vals = tl.load(a_copy_ptr + idx_plus_1, mask=mask_plus_1)
    
    # Compute d[i] = a[i] + a[i+1]
    d_vals = a_vals + a_plus_1_vals
    
    # Store d[i]
    tl.store(d_ptr + idx, d_vals, mask=mask)

def s1244_triton(a, b, c, d):
    n_elements = a.shape[0] - 1
    
    # Create read-only copy of a to handle WAR dependencies
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1244_kernel[grid](
        a, a_copy, b, c, d, n_elements, BLOCK_SIZE
    )