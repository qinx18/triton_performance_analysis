import torch
import triton
import triton.language as tl

@triton.jit
def s244_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load arrays
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        
        # Load a[i+1] values for the third operation
        next_offsets = current_offsets + 1
        next_mask = next_offsets < n_elements
        a_next_vals = tl.load(a_ptr + next_offsets, mask=next_mask, other=0.0)
        
        # First operation: a[i] = b[i] + c[i] * d[i]
        a_vals = b_vals + c_vals * d_vals
        tl.store(a_ptr + current_offsets, a_vals, mask=mask)
        
        # Second operation: b[i] = c[i] + b[i]
        b_new_vals = c_vals + b_vals
        tl.store(b_ptr + current_offsets, b_new_vals, mask=mask)
        
        # Third operation: a[i+1] = b[i] + a[i+1] * d[i]
        a_next_new = b_new_vals + a_next_vals * d_vals
        tl.store(a_ptr + next_offsets, a_next_new, mask=next_mask)

def s244_triton(a, b, c, d):
    n_elements = a.shape[0] - 1
    BLOCK_SIZE = 256
    
    s244_kernel[(1,)](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )