import triton
import triton.language as tl
import torch

@triton.jit
def s274_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load input data
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        e_vals = tl.load(e_ptr + current_offsets, mask=mask)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        
        # Compute a[i] = c[i] + e[i] * d[i]
        a_vals = c_vals + e_vals * d_vals
        
        # Apply conditional logic
        positive_mask = a_vals > 0.0
        
        # For positive values: b[i] = a[i] + b[i]
        new_b_vals = tl.where(positive_mask, a_vals + b_vals, b_vals)
        
        # For non-positive values: a[i] = d[i] * e[i]
        new_a_vals = tl.where(positive_mask, a_vals, d_vals * e_vals)
        
        # Store results
        tl.store(a_ptr + current_offsets, new_a_vals, mask=mask)
        tl.store(b_ptr + current_offsets, new_b_vals, mask=mask)

def s274_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    s274_kernel[(1,)](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )