import torch
import triton
import triton.language as tl

@triton.jit
def s1244_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        mask_next = (current_offsets + 1) < (n_elements + 1)
        
        # Load b[i], c[i]
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute a[i] = b[i] + c[i] * c[i] + b[i]*b[i] + c[i]
        a_vals = b_vals + c_vals * c_vals + b_vals * b_vals + c_vals
        
        # Store a[i]
        tl.store(a_ptr + current_offsets, a_vals, mask=mask)
        
        # Load a[i+1] for d[i] computation
        a_next_vals = tl.load(a_ptr + current_offsets + 1, mask=mask_next, other=0.0)
        
        # Compute d[i] = a[i] + a[i+1]
        d_vals = a_vals + a_next_vals
        
        # Store d[i]
        tl.store(d_ptr + current_offsets, d_vals, mask=mask)

def s1244_triton(a, b, c, d):
    n_elements = a.shape[0] - 1
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1244_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )