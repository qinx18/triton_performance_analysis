import torch
import triton
import triton.language as tl

@triton.jit
def s1244_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Mask for first computation (i < n_elements-1)
    mask1 = (block_start + offsets) < (n_elements - 1)
    
    # Load data for first computation
    b_vals = tl.load(b_ptr + block_start + offsets, mask=mask1)
    c_vals = tl.load(c_ptr + block_start + offsets, mask=mask1)
    
    # Compute a[i] = b[i] + c[i] * c[i] + b[i]*b[i] + c[i]
    a_vals = b_vals + c_vals * c_vals + b_vals * b_vals + c_vals
    
    # Store a[i]
    tl.store(a_ptr + block_start + offsets, a_vals, mask=mask1)
    
    # Load a[i+1] for second computation
    a_next_vals = tl.load(a_ptr + block_start + offsets + 1, mask=mask1)
    
    # Compute d[i] = a[i] + a[i+1]
    d_vals = a_vals + a_next_vals
    
    # Store d[i]
    tl.store(d_ptr + block_start + offsets, d_vals, mask=mask1)

def s1244_triton(a, b, c, d):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1244_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )