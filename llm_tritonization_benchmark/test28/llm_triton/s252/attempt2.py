import triton
import triton.language as tl
import torch

@triton.jit
def s252_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel cannot be parallelized due to the sequential dependency t = s
    # We need to process elements sequentially
    
    # Process all elements in sequential blocks
    t = 0.0
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        # Calculate how many elements to process in this block
        remaining = n_elements - block_start
        current_block_size = tl.minimum(BLOCK_SIZE, remaining)
        
        # Load elements for this block
        current_offsets = block_start + offsets
        mask = offsets < current_block_size
        
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process each element sequentially within the block
        for i in tl.static_range(BLOCK_SIZE):
            if i < current_block_size:
                s = b_vals[i] * c_vals[i]
                a_val = s + t
                
                # Store the result
                element_offset = block_start + i
                tl.store(a_ptr + element_offset, a_val)
                
                t = s

def s252_triton(a, b, c):
    N = a.shape[0]
    
    # Use a small block size since we need sequential processing
    BLOCK_SIZE = 1024
    
    # Launch kernel with single program instance to maintain sequential order
    grid = (1,)
    
    s252_kernel[grid](
        a, b, c,
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )