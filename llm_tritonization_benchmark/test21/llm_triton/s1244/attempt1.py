import triton
import triton.language as tl
import torch

@triton.jit
def s1244_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < (n - 1)
    
    # Load values
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    
    # Compute a[i] = b[i] + c[i] * c[i] + b[i]*b[i] + c[i]
    a_vals = b_vals + c_vals * c_vals + b_vals * b_vals + c_vals
    
    # Store a[i]
    tl.store(a_ptr + indices, a_vals, mask=mask)
    
    # Load a[i] from copy and a[i+1] from copy for d[i] computation
    a_i_vals = tl.load(a_copy_ptr + indices, mask=mask)
    a_i_plus_1_vals = tl.load(a_copy_ptr + indices + 1, mask=mask)
    
    # Compute d[i] = a[i] + a[i+1]
    d_vals = a_i_vals + a_i_plus_1_vals
    
    # Store d[i]
    tl.store(d_ptr + indices, d_vals, mask=mask)

def s1244_triton(a, b, c, d):
    n = a.shape[0]
    
    # Create read-only copy for WAR dependency handling
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n - 1, BLOCK_SIZE),)
    
    s1244_kernel[grid](
        a, a_copy, b, c, d, n,
        BLOCK_SIZE=BLOCK_SIZE
    )