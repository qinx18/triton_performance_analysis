import torch
import triton
import triton.language as tl

@triton.jit
def s212_kernel(a_ptr, b_ptr, c_ptr, d_ptr, a_copy_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load current elements
        a_vals = tl.load(a_copy_ptr + current_offsets, mask=mask)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        
        # Load a[i+1] values (from read-only copy)
        a_plus1_offsets = current_offsets + 1
        a_plus1_mask = a_plus1_offsets < (n_elements + 1)
        a_plus1_vals = tl.load(a_copy_ptr + a_plus1_offsets, mask=a_plus1_mask)
        
        # Perform computations
        new_a = a_vals * c_vals
        new_b = b_vals + a_plus1_vals * d_vals
        
        # Store results
        tl.store(a_ptr + current_offsets, new_a, mask=mask)
        tl.store(b_ptr + current_offsets, new_b, mask=mask)

def s212_triton(a, b, c, d):
    n_elements = a.shape[0] - 1
    
    # Create read-only copy to handle WAR dependency
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s212_kernel[grid](
        a, b, c, d, a_copy, n_elements, BLOCK_SIZE
    )