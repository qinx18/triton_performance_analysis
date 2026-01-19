import torch
import triton
import triton.language as tl

@triton.jit
def s1112_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process blocks in reverse order to maintain the reverse iteration pattern
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    for block_idx in range(num_blocks):
        # Calculate reverse block start
        reverse_block_idx = num_blocks - 1 - block_idx
        block_start = reverse_block_idx * BLOCK_SIZE
        
        # Calculate indices for this block (in reverse order within block)
        block_offsets = block_start + (BLOCK_SIZE - 1 - offsets)
        
        # Create mask for valid elements
        mask = (block_offsets >= 0) & (block_offsets < n_elements)
        
        # Load b values
        b_vals = tl.load(b_ptr + block_offsets, mask=mask)
        
        # Compute a[i] = b[i] + 1.0
        a_vals = b_vals + 1.0
        
        # Store results
        tl.store(a_ptr + block_offsets, a_vals, mask=mask)

def s1112_triton(a, b):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch single thread block since we need to maintain reverse order
    grid = (1,)
    s1112_kernel[grid](a, b, N, BLOCK_SIZE=BLOCK_SIZE)