import triton
import triton.language as tl
import torch

@triton.jit
def s243_kernel(
    a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        e_vals = tl.load(e_ptr + current_offsets, mask=mask)
        
        # Load a[i+1] values from copy (need to handle boundary)
        next_offsets = current_offsets + 1
        next_mask = next_offsets < (n_elements + 1)
        a_next_vals = tl.load(a_copy_ptr + next_offsets, mask=next_mask)
        
        # First computation: a[i] = b[i] + c[i] * d[i]
        a_vals = b_vals + c_vals * d_vals
        tl.store(a_ptr + current_offsets, a_vals, mask=mask)
        
        # Second computation: b[i] = a[i] + d[i] * e[i]
        b_vals = a_vals + d_vals * e_vals
        tl.store(b_ptr + current_offsets, b_vals, mask=mask)
        
        # Third computation: a[i] = b[i] + a[i+1] * d[i]
        a_vals = b_vals + a_next_vals * d_vals
        tl.store(a_ptr + current_offsets, a_vals, mask=mask)

def s243_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1  # Loop goes to LEN_1D-1
    BLOCK_SIZE = 256
    
    # Create read-only copy of array 'a' to handle WAR dependencies
    a_copy = a.clone()
    
    # Launch kernel
    s243_kernel[1,](
        a, a_copy, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )