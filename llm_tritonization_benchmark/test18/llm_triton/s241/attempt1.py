import triton
import triton.language as tl
import torch

@triton.jit
def s241_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load arrays
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        
        # Load a_copy for reading (immutable)
        a_copy_vals = tl.load(a_copy_ptr + current_offsets, mask=mask)
        
        # Load a_copy[i+1] for the second computation
        next_offsets = current_offsets + 1
        next_mask = next_offsets < n_elements
        a_copy_next = tl.load(a_copy_ptr + next_offsets, mask=next_mask)
        
        # First computation: a[i] = b[i] * c[i] * d[i]
        a_new = b_vals * c_vals * d_vals
        
        # Store new a values
        tl.store(a_ptr + current_offsets, a_new, mask=mask)
        
        # Second computation: b[i] = a[i] * a[i+1] * d[i]
        # Use the newly computed a values and a_copy for a[i+1]
        b_new = a_new * a_copy_next * d_vals
        
        # Store new b values
        tl.store(b_ptr + current_offsets, b_new, mask=mask)

def s241_triton(a, b, c, d):
    n = a.shape[0]
    n_elements = n - 1  # Loop goes to LEN_1D-1
    
    # Create read-only copy of array a to handle WAR dependencies
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s241_kernel[grid](
        a, a_copy, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )