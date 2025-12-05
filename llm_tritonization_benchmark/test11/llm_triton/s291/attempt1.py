import torch
import triton
import triton.language as tl

@triton.jit
def s291_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute offsets
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process blocks sequentially to maintain dependencies
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load current block of b
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        
        # For each element in the block, we need the previous element
        # Handle first element specially (uses last element of array)
        if block_start == 0:
            # First element uses b[n-1]
            b_prev_first = tl.load(b_ptr + (n_elements - 1))
            
            # For other elements in first block, use elements from same block
            for i in range(BLOCK_SIZE):
                if block_start + i >= n_elements:
                    break
                    
                current_offset = block_start + i
                b_current = tl.load(b_ptr + current_offset)
                
                if i == 0:
                    b_prev = b_prev_first
                else:
                    b_prev = tl.load(b_ptr + (current_offset - 1))
                
                result = (b_current + b_prev) * 0.5
                tl.store(a_ptr + current_offset, result)
        else:
            # For other blocks, previous element is always current_offset - 1
            for i in range(BLOCK_SIZE):
                if block_start + i >= n_elements:
                    break
                    
                current_offset = block_start + i
                b_current = tl.load(b_ptr + current_offset)
                b_prev = tl.load(b_ptr + (current_offset - 1))
                
                result = (b_current + b_prev) * 0.5
                tl.store(a_ptr + current_offset, result)

def s291_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch kernel with single program since we need sequential processing
    grid = (1,)
    s291_kernel[grid](
        a, b,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a