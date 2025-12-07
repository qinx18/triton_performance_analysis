import torch
import triton
import triton.language as tl

@triton.jit
def s124_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Process elements sequentially to maintain dependency on j
    j = tl.full([1], -1, dtype=tl.int32)
    
    # Process in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_block_size = min(BLOCK_SIZE, n_elements - block_start)
        
        # Create offsets once
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < current_block_size
        current_offsets = block_start + offsets
        
        # Load block data
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        d_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        e_vals = tl.load(e_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process each element in block sequentially
        for i in range(current_block_size):
            j_val = j[0] + 1
            j = tl.full([1], j_val, dtype=tl.int32)
            
            # Compute result based on condition
            if b_vals[i] > 0.0:
                result = b_vals[i] + d_vals[i] * e_vals[i]
            else:
                result = c_vals[i] + d_vals[i] * e_vals[i]
            
            # Store at position j
            j_offset = tl.full([1], j_val, dtype=tl.int32)
            tl.store(a_ptr + j_offset, result)

def s124_triton(a, b, c, d, e):
    n_elements = b.numel()
    
    # Use single thread for sequential processing
    BLOCK_SIZE = 64
    grid = (1,)
    
    s124_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a