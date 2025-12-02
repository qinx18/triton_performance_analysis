import torch
import triton
import triton.language as tl

@triton.jit
def s241_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process elements in chunks
    for chunk_start in range(0, n_elements - 1, BLOCK_SIZE):
        current_offsets = chunk_start + offsets
        mask = current_offsets < (n_elements - 1)
        
        # Load current values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        
        # First computation: a[i] = b[i] * c[i] * d[i]
        a_new = b_vals * c_vals * d_vals
        tl.store(a_ptr + current_offsets, a_new, mask=mask)
        
        # Load a[i+1] for second computation
        next_offsets = current_offsets + 1
        next_mask = next_offsets < n_elements
        a_next = tl.load(a_ptr + next_offsets, mask=next_mask)
        
        # Second computation: b[i] = a[i] * a[i+1] * d[i]
        # Use 0 for out-of-bounds a[i+1] values
        a_next = tl.where(next_mask, a_next, 0.0)
        b_new = a_new * a_next * d_vals
        tl.store(b_ptr + current_offsets, b_new, mask=mask)

def s241_triton(a, b, c, d):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch kernel with single thread block to handle dependencies
    grid = (1,)
    
    s241_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )