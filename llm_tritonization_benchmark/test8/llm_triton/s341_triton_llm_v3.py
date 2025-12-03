import torch
import triton
import triton.language as tl

@triton.jit
def s341_kernel(b_ptr, a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process elements in blocks
    write_idx = 0
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load block of b values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        # Check which values are positive
        pos_mask = b_vals > 0.0
        
        # Count positive values in this block
        pos_count = tl.sum(pos_mask.to(tl.int32))
        
        # For each positive value, write it to output array
        for i in range(BLOCK_SIZE):
            if block_start + i < n_elements:
                val = tl.load(b_ptr + block_start + i)
                if val > 0.0:
                    tl.store(a_ptr + write_idx, val)
                    write_idx += 1

def s341_triton(a, b):
    n_elements = b.shape[0]
    BLOCK_SIZE = 256
    
    # Reset output array
    a.zero_()
    
    # Launch kernel with single thread block to maintain sequential ordering
    grid = (1,)
    s341_kernel[grid](b, a, n_elements, BLOCK_SIZE)
    
    return a