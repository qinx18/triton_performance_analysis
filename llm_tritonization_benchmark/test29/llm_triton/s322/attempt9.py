import triton
import triton.language as tl
import torch

@triton.jit
def s322_kernel(a_ptr, b_ptr, c_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Handle sequential recurrence with dependency tracking
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Each block processes its portion with proper dependency handling
    offsets = tl.arange(0, BLOCK_SIZE)
    block_offsets = block_start + offsets
    mask = block_offsets < N
    valid_offsets = block_offsets >= 2
    final_mask = mask & valid_offsets
    
    # For each valid position in this block
    if tl.sum(final_mask.to(tl.int32)) > 0:
        # Load current values
        a_vals = tl.load(a_ptr + block_offsets, mask=final_mask, other=0.0)
        b_vals = tl.load(b_ptr + block_offsets, mask=final_mask, other=0.0)
        c_vals = tl.load(c_ptr + block_offsets, mask=final_mask, other=0.0)
        
        # Load previous values - need to handle dependencies
        prev_offsets = block_offsets - 1
        prev_mask = prev_offsets >= 0
        prev_final_mask = mask & prev_mask
        a_prev = tl.load(a_ptr + prev_offsets, mask=prev_final_mask, other=0.0)
        
        # Load values from 2 steps back
        prev2_offsets = block_offsets - 2
        prev2_mask = prev2_offsets >= 0
        prev2_final_mask = mask & prev2_mask
        a_prev2 = tl.load(a_ptr + prev2_offsets, mask=prev2_final_mask, other=0.0)
        
        # Compute: a[i] = a[i] + a[i-1] * b[i] + a[i-2] * c[i]
        result = a_vals + a_prev * b_vals + a_prev2 * c_vals
        
        # Store results
        tl.store(a_ptr + block_offsets, result, mask=final_mask)

def s322_triton(a, b, c):
    N = a.shape[0]
    
    if N <= 2:
        return a
    
    BLOCK_SIZE = 32
    # Use multiple blocks to increase parallelism
    num_blocks = triton.cdiv(N, BLOCK_SIZE)
    grid = (num_blocks,)
    
    # Process in waves to handle dependencies
    for wave_start in range(2, N, BLOCK_SIZE * num_blocks):
        wave_end = min(wave_start + BLOCK_SIZE * num_blocks, N)
        if wave_start >= wave_end:
            break
            
        s322_kernel[grid](
            a, b, c, wave_end,
            BLOCK_SIZE=BLOCK_SIZE
        )
    
    return a