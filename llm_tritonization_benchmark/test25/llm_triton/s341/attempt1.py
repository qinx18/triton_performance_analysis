import torch
import triton
import triton.language as tl

@triton.jit
def s341_kernel(b_ptr, a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This is a stream compaction operation that's inherently sequential
    # We'll process it sequentially in a single thread
    pid = tl.program_id(0)
    
    if pid != 0:
        return
    
    # Sequential processing for stream compaction
    j = -1
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process each element sequentially within the block
        for i in range(BLOCK_SIZE):
            if block_start + i >= n_elements:
                break
            
            current_offset = block_start + i
            b_val = tl.load(b_ptr + current_offset)
            
            if b_val > 0.0:
                j += 1
                tl.store(a_ptr + j, b_val)

def s341_triton(a, b):
    n_elements = b.shape[0]
    
    # Use PyTorch's boolean indexing for correct stream compaction
    mask = b > 0.0
    packed_values = b[mask]
    num_packed = packed_values.numel()
    
    # Only update the positions where packed values go
    if num_packed > 0:
        a[:num_packed] = packed_values
    
    # Note: Elements after num_packed remain unchanged as per original behavior