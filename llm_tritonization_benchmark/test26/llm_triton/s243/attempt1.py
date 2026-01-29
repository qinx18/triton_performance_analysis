import triton
import triton.language as tl
import torch

@triton.jit
def s243_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, a_copy_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values for current iteration
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
        e_vals = tl.load(e_ptr + current_offsets, mask=mask, other=0.0)
        
        # Load a[i+1] from the read-only copy
        next_offsets = current_offsets + 1
        next_mask = next_offsets < (n_elements + 1)
        a_next_vals = tl.load(a_copy_ptr + next_offsets, mask=next_mask, other=0.0)
        
        # First statement: a[i] = b[i] + c[i] * d[i]
        a_vals = b_vals + c_vals * d_vals
        tl.store(a_ptr + current_offsets, a_vals, mask=mask)
        
        # Second statement: b[i] = a[i] + d[i] * e[i]
        b_vals = a_vals + d_vals * e_vals
        tl.store(b_ptr + current_offsets, b_vals, mask=mask)
        
        # Third statement: a[i] = b[i] + a[i+1] * d[i]
        a_vals = b_vals + a_next_vals * d_vals
        tl.store(a_ptr + current_offsets, a_vals, mask=mask)

def s243_triton(a, b, c, d, e):
    n = a.shape[0]
    n_elements = n - 1  # Loop runs from 0 to LEN_1D-2
    
    # Create read-only copy of 'a' to handle WAR dependency
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s243_kernel[grid](
        a, b, c, d, e, a_copy,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )