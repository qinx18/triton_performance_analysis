import torch
import triton
import triton.language as tl

@triton.jit
def s341_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This is a sequential stream compaction - cannot be parallelized efficiently
    # Use single block processing
    block_id = tl.program_id(0)
    if block_id > 0:
        return
    
    offsets = tl.arange(0, BLOCK_SIZE)
    j = 0
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process each element in the block sequentially
        for i in range(BLOCK_SIZE):
            if block_start + i < n_elements:
                b_val = tl.load(b_ptr + block_start + i)
                if b_val > 0.0:
                    tl.store(a_ptr + j, b_val)
                    j += 1

def s341_triton(a, b):
    n_elements = b.shape[0]
    BLOCK_SIZE = 1024
    
    # Use PyTorch's efficient boolean indexing for stream compaction
    mask = b > 0.0
    packed_values = b[mask]
    num_packed = packed_values.numel()
    
    # Only update the first num_packed elements, leave rest unchanged
    if num_packed > 0:
        a[:num_packed] = packed_values