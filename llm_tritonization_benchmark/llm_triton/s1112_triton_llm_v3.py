import torch
import triton
import triton.language as tl

@triton.jit
def s1112_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Define offsets once at kernel start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process blocks in reverse order to match original loop behavior
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    
    for block_idx in range(num_blocks):
        # Calculate reverse block starting position
        reverse_block_idx = num_blocks - 1 - block_idx
        block_start = reverse_block_idx * BLOCK_SIZE
        
        # Calculate current offsets for this block
        current_offsets = block_start + offsets
        
        # Create mask for valid elements
        mask = current_offsets < n_elements
        
        # Load b values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        
        # Compute a[i] = b[i] + 1.0
        a_vals = b_vals + 1.0
        
        # Store results
        tl.store(a_ptr + current_offsets, a_vals, mask=mask)

def s1112_triton(a, b):
    n_elements = a.numel()
    
    BLOCK_SIZE = 256
    grid = (tl.cdiv(n_elements, BLOCK_SIZE),)
    
    s1112_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )