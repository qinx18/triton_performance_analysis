import torch
import triton
import triton.language as tl

@triton.jit
def s1221_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Each program handles one stream of the strided pattern
    stream_id = tl.program_id(0)
    
    # Calculate starting position for this stream (starts at 4 + stream_id)
    start_pos = 4 + stream_id
    
    if start_pos >= n:
        return
    
    # Load initial value for this stream
    running_sum = tl.load(b_ptr + stream_id)
    
    # Process elements in this stream with stride 4
    pos = start_pos
    while pos < n:
        # Load addend from array a
        addend = tl.load(a_ptr + pos)
        
        # Update running sum
        running_sum += addend
        
        # Store result
        tl.store(b_ptr + pos, running_sum)
        
        # Move to next element in this stream
        pos += 4

def s1221_triton(a, b):
    n = a.shape[0]
    
    if n <= 4:
        return
    
    # Launch 4 programs, one for each stream
    grid = (4,)
    BLOCK_SIZE = 128
    
    s1221_kernel[grid](
        a, b, n,
        BLOCK_SIZE=BLOCK_SIZE
    )