import triton
import triton.language as tl
import torch

@triton.jit
def s241_kernel(
    a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        next_offsets = current_offsets + 1
        
        mask = current_offsets < n_elements
        next_mask = next_offsets < n_elements
        
        # Load data for first statement: a[i] = b[i] * c[i] * d[i]
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        
        # Compute and store a[i]
        new_a_vals = b_vals * c_vals * d_vals
        tl.store(a_ptr + current_offsets, new_a_vals, mask=mask)
        
        # Load data for second statement: b[i] = a[i] * a[i+1] * d[i]
        a_curr_vals = tl.load(a_copy_ptr + current_offsets, mask=mask)
        a_next_vals = tl.load(a_copy_ptr + next_offsets, mask=next_mask)
        
        # Compute and store b[i]
        new_b_vals = a_curr_vals * a_next_vals * d_vals
        tl.store(b_ptr + current_offsets, new_b_vals, mask=mask)

def s241_triton(a, b, c, d):
    n_elements = a.shape[0] - 1
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s241_kernel[grid](
        a, a_copy, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )