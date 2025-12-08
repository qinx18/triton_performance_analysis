import torch
import triton
import triton.language as tl

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
        
        # Load values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
        a_next_vals = tl.load(a_copy_ptr + next_offsets, mask=next_mask, other=0.0)
        
        # First computation: a[i] = b[i] * c[i] * d[i]
        a_new = b_vals * c_vals * d_vals
        
        # Store a values
        tl.store(a_ptr + current_offsets, a_new, mask=mask)
        
        # Second computation: b[i] = a[i] * a[i+1] * d[i]
        b_new = a_new * a_next_vals * d_vals
        
        # Store b values
        tl.store(b_ptr + current_offsets, b_new, mask=mask)

def s241_triton(a, b, c, d):
    n_elements = a.shape[0] - 1
    
    # Create read-only copy for WAR dependency handling
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s241_kernel[grid](
        a, a_copy, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )