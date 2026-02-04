import torch
import triton
import triton.language as tl

@triton.jit
def s1221_kernel(a_ptr, b_ptr, n, stream_id: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Each program handles a different block of elements within one stream
    block_id = tl.program_id(0)
    
    # Calculate starting position for this stream
    stream_start = 4 + stream_id
    
    # Calculate block start within this stream
    block_start_in_stream = block_id * BLOCK_SIZE
    
    # Calculate actual positions in the array
    block_offsets = tl.arange(0, BLOCK_SIZE)
    positions = stream_start + block_start_in_stream * 4 + block_offsets * 4
    
    # Mask for valid positions
    mask = positions < n
    
    # Load the addends from array a
    addends = tl.load(a_ptr + positions, mask=mask, other=0.0)
    
    # Compute cumulative sum within this block
    running_sum = addends
    for i in range(1, BLOCK_SIZE):
        shift_mask = block_offsets >= i
        shifted_sum = tl.where(shift_mask, running_sum, 0.0)
        shifted_sum = tl.broadcast_to(shifted_sum, (BLOCK_SIZE,))
        prev_sum = tl.where(block_offsets == i, 
                           tl.sum(tl.where(block_offsets < i, running_sum, 0.0)),
                           0.0)
        running_sum = tl.where(block_offsets == i, prev_sum + addends, running_sum)
    
    # Add the initial value from b[stream_id]
    initial_val = tl.load(b_ptr + stream_id)
    
    # Add contribution from previous blocks
    if block_id > 0:
        # Load the last value from previous block
        prev_last_pos = stream_start + (block_id - 1) * BLOCK_SIZE * 4 + (BLOCK_SIZE - 1) * 4
        if prev_last_pos < n:
            prev_contribution = tl.load(b_ptr + prev_last_pos) - initial_val
            running_sum += prev_contribution
    
    final_values = initial_val + running_sum
    
    # Store results
    tl.store(b_ptr + positions, final_values, mask=mask)

def s1221_triton(a, b):
    n = a.shape[0]
    
    if n <= 4:
        return
    
    BLOCK_SIZE = 64
    
    # Process each stream sequentially
    for stream in range(4):
        stream_start = 4 + stream
        stream_length = (n - stream_start + 3) // 4  # Number of elements in this stream
        
        if stream_length <= 0:
            continue
            
        # Number of blocks needed for this stream
        num_blocks = triton.cdiv(stream_length, BLOCK_SIZE)
        
        # Process blocks sequentially within each stream to maintain dependencies
        for block in range(num_blocks):
            grid = (1,)
            
            # Create a kernel for this specific block
            @triton.jit
            def stream_block_kernel(a_ptr, b_ptr, n, block_offset: tl.constexpr, 
                                   stream_id: tl.constexpr, BLOCK_SIZE: tl.constexpr):
                # Calculate starting position for this stream and block
                stream_start = 4 + stream_id
                block_start_in_stream = block_offset
                
                # Calculate actual positions in the array
                block_offsets = tl.arange(0, BLOCK_SIZE)
                positions = stream_start + block_start_in_stream * 4 + block_offsets * 4
                
                # Mask for valid positions
                mask = positions < n
                
                # Load the addends from array a
                addends = tl.load(a_ptr + positions, mask=mask, other=0.0)
                
                # Get initial value for cumsum
                if block_offset == 0:
                    initial_val = tl.load(b_ptr + stream_id)
                    cumsum_base = 0.0
                else:
                    # Get the last computed value from previous iteration
                    prev_pos = positions[0] - 4
                    cumsum_base = tl.load(b_ptr + prev_pos)
                    initial_val = 0.0
                
                # Compute prefix sum of addends within this block
                running_sum = addends
                for i in range(1, BLOCK_SIZE):
                    prev_val = tl.where(block_offsets == (i-1), running_sum, 0.0)
                    prev_sum = tl.sum(prev_val)
                    current_mask = block_offsets >= i
                    running_sum = tl.where(current_mask, 
                                         tl.where(block_offsets == i, prev_sum + addends, running_sum),
                                         running_sum)
                
                # Add base values
                final_values = cumsum_base + initial_val + running_sum
                
                # Store results
                tl.store(b_ptr + positions, final_values, mask=mask)
            
            stream_block_kernel[grid](a, b, n, block, stream, BLOCK_SIZE=BLOCK_SIZE)