import torch
import triton
import triton.language as tl

@triton.jit
def s1221_kernel(a_ptr, b_ptr, initial_val_ptr, stream_indices_ptr, n_stream: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_stream
    
    # Load initial value as scalar
    initial_val = tl.load(initial_val_ptr)
    
    # Load a values for this block
    a_vals = tl.load(a_ptr + tl.load(stream_indices_ptr + indices, mask=mask, other=0), mask=mask, other=0.0)
    
    # Compute prefix sum within block
    cumsum = a_vals
    for step in range(1, BLOCK_SIZE):
        prev_val = tl.where(offsets == step - 1, cumsum, 0.0)
        prev_sum = tl.sum(prev_val)
        shift_mask = offsets >= step
        cumsum = tl.where(shift_mask & mask, cumsum + prev_sum, cumsum)
    
    # Add initial value
    result = cumsum + initial_val
    
    # Store results back to b
    tl.store(b_ptr + tl.load(stream_indices_ptr + indices, mask=mask, other=0), result, mask=mask)

def s1221_triton(a, b):
    n = a.shape[0]
    
    if n <= 4:
        return
    
    BLOCK_SIZE = 256
    
    # Process each stream independently
    for stream in range(4):
        # Get indices for this stream starting from position 4+stream
        stream_start = 4 + stream
        stream_indices = torch.arange(stream_start, n, 4, device=a.device, dtype=torch.long)
        
        if len(stream_indices) == 0:
            continue
        
        n_stream = len(stream_indices)
        num_blocks = triton.cdiv(n_stream, BLOCK_SIZE)
        
        # Create tensor for initial value
        initial_val_tensor = torch.tensor([b[stream]], device=a.device, dtype=a.dtype)
        
        if num_blocks > 0:
            # Launch kernel for each block of this stream
            grid = (num_blocks,)
            s1221_kernel[grid](a, b, initial_val_tensor, stream_indices, n_stream, BLOCK_SIZE=BLOCK_SIZE)
            
            # Handle inter-block prefix sum
            for block in range(1, num_blocks):
                start_idx = block * BLOCK_SIZE
                if start_idx < n_stream:
                    # Get cumulative sum from previous block's last element
                    prev_last_idx = stream_indices[start_idx - 1]
                    prev_cumsum = b[prev_last_idx]
                    
                    # Add to all elements in current block
                    end_idx = min((block + 1) * BLOCK_SIZE, n_stream)
                    current_indices = stream_indices[start_idx:end_idx]
                    b[current_indices] += prev_cumsum