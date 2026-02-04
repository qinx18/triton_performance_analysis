import torch
import triton
import triton.language as tl

@triton.jit
def s1221_kernel(a_ptr, b_ptr, stream_indices_ptr, addends_ptr, n_stream: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    
    offsets = tl.arange(0, BLOCK_SIZE)
    positions = block_id * BLOCK_SIZE + offsets
    
    mask = positions < n_stream
    
    # Load addends for this block
    addends = tl.load(addends_ptr + positions, mask=mask, other=0.0)
    
    # Compute prefix sum within block
    cumsum = addends
    for step in range(1, BLOCK_SIZE):
        shift_mask = offsets >= step
        prev_vals = tl.where(offsets == step - 1, cumsum, 0.0)
        prev_sum = tl.sum(prev_vals)
        cumsum = tl.where(shift_mask & (offsets == step), addends + prev_sum, cumsum)
    
    # Store prefix sums back
    tl.store(addends_ptr + positions, cumsum, mask=mask)

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
        
        # Extract addends for this stream
        addends = a[stream_indices].clone()
        
        n_stream = len(stream_indices)
        num_blocks = triton.cdiv(n_stream, BLOCK_SIZE)
        
        if num_blocks > 0:
            # Compute prefix sum of addends using Triton kernel
            grid = (num_blocks,)
            s1221_kernel[grid](a, b, stream_indices, addends, n_stream, BLOCK_SIZE=BLOCK_SIZE)
            
            # Handle inter-block dependencies
            for block in range(1, num_blocks):
                start_idx = block * BLOCK_SIZE
                end_idx = min((block + 1) * BLOCK_SIZE, n_stream)
                
                if start_idx < n_stream:
                    prev_cumsum = addends[start_idx - 1].item()
                    addends[start_idx:end_idx] += prev_cumsum
        
        # Add initial value b[stream] and store results
        if n_stream > 0:
            initial_val = b[stream].item()
            final_values = initial_val + addends
            b[stream_indices] = final_values