import triton
import triton.language as tl
import torch

@triton.jit
def s1244_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < (n - 1)
    
    # Load data
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    a_next_vals = tl.load(a_copy_ptr + idx + 1, mask=mask)
    
    # First computation: a[i] = b[i] + c[i] * c[i] + b[i]*b[i] + c[i]
    a_vals = b_vals + c_vals * c_vals + b_vals * b_vals + c_vals
    
    # Second computation: d[i] = a[i] + a[i+1]
    d_vals = a_vals + a_next_vals
    
    # Store results
    tl.store(a_ptr + idx, a_vals, mask=mask)
    tl.store(d_ptr + idx, d_vals, mask=mask)

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