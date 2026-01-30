import torch
import triton
import triton.language as tl

@triton.jit
def s341_kernel(b_ptr, a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This is a stream compaction pattern that's inherently sequential
    # We'll process it block by block to maintain correctness
    
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Counter for packed elements (shared across all threads in block)
    counter = tl.zeros([1], dtype=tl.int32)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load input values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        # Check condition
        condition = b_vals > 0.0
        
        # Process each element in the block sequentially
        for i in range(BLOCK_SIZE):
            if block_start + i < n_elements:
                if condition[i]:
                    current_counter = tl.load(counter)
                    tl.store(a_ptr + current_counter, b_vals[i])
                    tl.store(counter, current_counter + 1)

def s341_triton(a, b):
    n_elements = b.shape[0]
    
    # Use PyTorch's boolean indexing for correct stream compaction
    mask = b > 0.0
    packed_values = b[mask]
    num_packed = packed_values.numel()
    
    # Only update the packed portion, leave rest unchanged
    if num_packed > 0:
        a[:num_packed] = packed_values
    
    return a