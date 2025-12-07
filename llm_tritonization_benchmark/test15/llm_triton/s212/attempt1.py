import triton
import triton.language as tl
import torch

@triton.jit
def s212_kernel(
    a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load data
        a_vals = tl.load(a_copy_ptr + current_offsets, mask=mask)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        
        # Load a[i+1] values (read from copy)
        next_offsets = current_offsets + 1
        next_mask = mask & (next_offsets < (n_elements + 1))  # Allow reading a[i+1] within bounds
        a_next_vals = tl.load(a_copy_ptr + next_offsets, mask=next_mask)
        
        # Compute: a[i] *= c[i]
        new_a_vals = a_vals * c_vals
        
        # Compute: b[i] += a[i + 1] * d[i]
        new_b_vals = b_vals + a_next_vals * d_vals
        
        # Store results
        tl.store(a_ptr + current_offsets, new_a_vals, mask=mask)
        tl.store(b_ptr + current_offsets, new_b_vals, mask=mask)

def s212_triton(a, b, c, d):
    n_elements = a.shape[0] - 1  # Process LEN_1D-1 elements
    
    # Create read-only copy to handle WAR dependency
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s212_kernel[grid](
        a, a_copy, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )