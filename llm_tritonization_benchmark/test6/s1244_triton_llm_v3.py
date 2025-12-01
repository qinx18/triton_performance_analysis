import triton
import triton.language as tl
import torch

@triton.jit
def s1244_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # First pass: compute a[i] = b[i] + c[i] * c[i] + b[i]*b[i] + c[i]
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        
        a_vals = b_vals + c_vals * c_vals + b_vals * b_vals + c_vals
        tl.store(a_ptr + current_offsets, a_vals, mask=mask)
    
    # Second pass: compute d[i] = a[i] + a[i+1]
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        a_vals = tl.load(a_ptr + current_offsets, mask=mask)
        a_next_vals = tl.load(a_ptr + current_offsets + 1, mask=mask)
        
        d_vals = a_vals + a_next_vals
        tl.store(d_ptr + current_offsets, d_vals, mask=mask)

def s1244_triton(a, b, c, d):
    n_elements = a.shape[0] - 1
    BLOCK_SIZE = 256
    
    s1244_kernel[(1,)](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )