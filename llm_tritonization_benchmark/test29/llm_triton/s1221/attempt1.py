import torch
import triton
import triton.language as tl

@triton.jit
def s1221_kernel(b_ptr, a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This implements a strided prefix sum with stride 4
    # Each block handles one stream (stream 0, 1, 2, or 3)
    
    stream_id = tl.program_id(0)
    
    if stream_id >= 4:
        return
    
    # Starting index for this stream (4 + stream_id, 8 + stream_id, etc.)
    start_idx = 4 + stream_id
    
    if start_idx >= n_elements:
        return
    
    # Load initial value b[stream_id] 
    initial_val = tl.load(b_ptr + stream_id)
    
    # Calculate number of elements in this stream
    stream_length = (n_elements - start_idx + 3) // 4  # ceiling division
    
    if stream_length <= 0:
        return
    
    # Process elements in blocks
    offsets = tl.arange(0, BLOCK_SIZE)
    
    running_sum = 0.0
    
    for block_start in range(0, stream_length, BLOCK_SIZE):
        # Calculate actual indices for this block
        block_offsets = block_start + offsets
        actual_indices = start_idx + block_offsets * 4
        
        mask = (block_offsets < stream_length) & (actual_indices < n_elements)
        
        # Load a values for this block
        a_vals = tl.load(a_ptr + actual_indices, mask=mask, other=0.0)
        
        # Compute prefix sum within this block
        block_prefix = tl.cumsum(a_vals, axis=0)
        
        # Add the running sum from previous blocks
        final_vals = initial_val + running_sum + block_prefix
        
        # Store results
        tl.store(b_ptr + actual_indices, final_vals, mask=mask)
        
        # Update running sum for next block
        if block_start + BLOCK_SIZE < stream_length:
            running_sum = running_sum + tl.sum(a_vals)

def s1221_triton(a, b):
    n_elements = a.shape[0]
    
    if n_elements <= 4:
        return
    
    BLOCK_SIZE = 256
    
    # Launch 4 blocks, one for each stream
    grid = (4,)
    
    s1221_kernel[grid](
        b, a, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )