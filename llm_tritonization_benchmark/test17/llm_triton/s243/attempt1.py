import triton
import triton.language as tl
import torch

@triton.jit
def s243_kernel(
    a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Process sequentially to handle dependencies
    for block_start in range(0, n_elements, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load values for current block
        b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + offsets, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + offsets, mask=mask, other=0.0)
        e_vals = tl.load(e_ptr + offsets, mask=mask, other=0.0)
        
        # Load a[i+1] values from read-only copy
        next_offsets = offsets + 1
        next_mask = next_offsets < n_elements
        a_next_vals = tl.load(a_copy_ptr + next_offsets, mask=next_mask, other=0.0)
        
        # First statement: a[i] = b[i] + c[i] * d[i]
        a_vals = b_vals + c_vals * d_vals
        tl.store(a_ptr + offsets, a_vals, mask=mask)
        
        # Second statement: b[i] = a[i] + d[i] * e[i]
        b_vals = a_vals + d_vals * e_vals
        tl.store(b_ptr + offsets, b_vals, mask=mask)
        
        # Third statement: a[i] = b[i] + a[i+1] * d[i]
        # For the last element, don't update since i+1 would be out of bounds
        final_mask = mask & (offsets < (n_elements - 1))
        a_vals = b_vals + a_next_vals * d_vals
        tl.store(a_ptr + offsets, a_vals, mask=final_mask)

def s243_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1  # Process LEN_1D-1 elements
    
    # Create read-only copy of array a
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (1,)  # Single thread block for sequential processing
    
    s243_kernel[grid](
        a, a_copy, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )