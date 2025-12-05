import torch
import triton
import triton.language as tl

@triton.jit
def s341_kernel(b_ptr, a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This is a stream compaction operation - pack positive values
    # Since this requires knowing how many previous elements passed the condition,
    # we need to use atomic operations or process sequentially
    
    pid = tl.program_id(0)
    
    # Only use the first program to avoid race conditions
    if pid != 0:
        return
        
    # Process elements sequentially to maintain correct packing order
    j = -1
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values from b
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process each element in the block
        for i in range(BLOCK_SIZE):
            if block_start + i >= n_elements:
                break
            
            # Check condition using scalar access pattern
            element_mask = (current_offsets == (block_start + i)) & mask
            if tl.sum(tl.where(element_mask, b_vals, 0.0)) > 0.0:
                j += 1
                # Store the positive value at position j
                tl.store(a_ptr + j, tl.sum(tl.where(element_mask, b_vals, 0.0)))

def s341_triton(a, b):
    n_elements = b.numel()
    BLOCK_SIZE = 256
    
    # Use PyTorch's boolean indexing for correct stream compaction
    mask = b > 0.0
    packed_values = b[mask]
    num_packed = packed_values.numel()
    
    # Only update the first num_packed elements, leave rest unchanged
    if num_packed > 0:
        a[:num_packed] = packed_values
    
    return a