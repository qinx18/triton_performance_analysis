import triton
import triton.language as tl
import torch

@triton.jit
def s212_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, a_copy_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load current elements
        a_vals = tl.load(a_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        
        # Load a[i+1] from read-only copy
        next_offsets = current_offsets + 1
        next_mask = next_offsets < (n_elements + 1)
        a_next_vals = tl.load(a_copy_ptr + next_offsets, mask=next_mask)
        
        # Perform computations
        # a[i] *= c[i]
        new_a_vals = a_vals * c_vals
        
        # b[i] += a[i + 1] * d[i]
        new_b_vals = b_vals + a_next_vals * d_vals
        
        # Store results
        tl.store(a_ptr + current_offsets, new_a_vals, mask=mask)
        tl.store(b_ptr + current_offsets, new_b_vals, mask=mask)

def s212_triton(a, b, c, d):
    n_elements = a.shape[0] - 1  # Loop goes to LEN_1D-1
    
    # Create read-only copy for WAR dependency handling
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s212_kernel[grid](
        a, b, c, d, a_copy,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )