import triton
import triton.language as tl
import torch

@triton.jit
def s1281_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values
        a_vals = tl.load(a_ptr + current_offsets, mask=mask)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        e_vals = tl.load(e_ptr + current_offsets, mask=mask)
        
        # Compute x = b[i]*c[i] + a[i]*d[i] + e[i]
        x = b_vals * c_vals + a_vals * d_vals + e_vals
        
        # Store results
        # a[i] = x - 1.0
        tl.store(a_ptr + current_offsets, x - 1.0, mask=mask)
        # b[i] = x
        tl.store(b_ptr + current_offsets, x, mask=mask)

def s1281_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    grid = (1,)
    
    s1281_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )