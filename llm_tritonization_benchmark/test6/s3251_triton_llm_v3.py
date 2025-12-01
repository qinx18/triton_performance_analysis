import triton
import triton.language as tl
import torch

@triton.jit
def s3251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load current values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        e_vals = tl.load(e_ptr + current_offsets, mask=mask)
        a_vals = tl.load(a_ptr + current_offsets, mask=mask)
        
        # Compute operations
        # a[i+1] = b[i] + c[i]
        a_new = b_vals + c_vals
        
        # b[i] = c[i] * e[i]
        b_new = c_vals * e_vals
        
        # d[i] = a[i] * e[i]
        d_new = a_vals * e_vals
        
        # Store results
        tl.store(a_ptr + current_offsets + 1, a_new, mask=mask)
        tl.store(b_ptr + current_offsets, b_new, mask=mask)
        tl.store(d_ptr + current_offsets, d_new, mask=mask)

def s3251_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1
    BLOCK_SIZE = 256
    
    s3251_kernel[(1,)](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )