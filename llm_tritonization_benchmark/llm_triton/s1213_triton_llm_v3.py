import torch
import triton
import triton.language as tl

@triton.jit
def s1213_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Load b_ro (read-only copy) for the entire range we need
    b_ro = tl.zeros([BLOCK_SIZE + 2], dtype=tl.float32)
    
    # Process elements sequentially to handle dependencies
    for block_start in range(1, n_elements - 1, BLOCK_SIZE):
        block_end = min(block_start + BLOCK_SIZE, n_elements - 1)
        actual_block_size = block_end - block_start
        
        # Create offset arrays for current block
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = tl.arange(0, BLOCK_SIZE) < actual_block_size
        
        # Load required data with extended range for dependencies
        # Load b values from (block_start-1) to (block_start + BLOCK_SIZE)
        b_offsets_ext = (block_start - 1) + tl.arange(0, BLOCK_SIZE + 2)
        b_mask_ext = tl.arange(0, BLOCK_SIZE + 2) < (actual_block_size + 2)
        b_ro_vals = tl.load(b_ptr + b_offsets_ext, mask=b_mask_ext, other=0.0)
        
        # Load c and d values for current block
        c_vals = tl.load(c_ptr + offsets, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + offsets, mask=mask, other=0.0)
        
        # Load a[i+1] values (needed for second statement)
        a_next_offsets = offsets + 1
        a_next_mask = mask & (offsets + 1 < n_elements - 1)
        a_next_vals = tl.load(a_ptr + a_next_offsets, mask=a_next_mask, other=0.0)
        
        # Process each element in the block sequentially
        for i in range(actual_block_size):
            if block_start + i < n_elements - 1:
                idx = block_start + i
                
                # First statement: a[i] = b[i-1] + c[i]
                b_prev_val = tl.load(b_ptr + (idx - 1))
                c_val = tl.load(c_ptr + idx)
                a_val = b_prev_val + c_val
                tl.store(a_ptr + idx, a_val)
                
                # Second statement: b[i] = a[i+1] * d[i]
                a_next_val = tl.load(a_ptr + (idx + 1))
                d_val = tl.load(d_ptr + idx)
                b_val = a_next_val * d_val
                tl.store(b_ptr + idx, b_val)

def s1213_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # Use small block size due to sequential processing requirements
    BLOCK_SIZE = 32
    
    # Launch kernel with single program
    grid = (1,)
    
    s1213_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )