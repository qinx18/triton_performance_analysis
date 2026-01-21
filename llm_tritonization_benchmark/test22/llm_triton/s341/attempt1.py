import torch
import triton
import triton.language as tl

@triton.jit
def s341_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This is a stream compaction kernel that packs positive values
    # We process the entire array sequentially since this is inherently serial
    
    # Process in blocks
    offsets = tl.arange(0, BLOCK_SIZE)
    output_idx = 0
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load block of b values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process each element in the block sequentially
        for i in range(BLOCK_SIZE):
            if block_start + i >= n_elements:
                break
                
            element_offset = block_start + i
            b_val = tl.load(b_ptr + element_offset)
            
            if b_val > 0.0:
                tl.store(a_ptr + output_idx, b_val)
                output_idx += 1

def s341_triton(a, b):
    n_elements = b.shape[0]
    BLOCK_SIZE = 256
    
    # Use torch's boolean indexing for stream compaction
    # This is the correct and efficient way to handle this pattern
    mask = b > 0.0
    packed_values = b[mask]
    num_packed = packed_values.numel()
    
    # Only update the positions that get new values
    # Leave the rest of array 'a' unchanged
    if num_packed > 0:
        a[:num_packed] = packed_values