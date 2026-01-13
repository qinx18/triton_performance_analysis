import torch
import triton
import triton.language as tl

@triton.jit
def s141_kernel(
    bb_ptr,
    flat_2d_array_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    i = pid
    
    if i >= N:
        return
    
    # Calculate initial k: k = (i+1) * ((i+1) - 1) / 2 + (i+1) - 1
    # This simplifies to: k = i * (i + 1) / 2 + i = i * (i + 3) / 2
    k = i * (i + 3) // 2
    
    # Process the inner loop: for j in range(i, N)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    j_block_start = i
    while j_block_start < N:
        current_j_offsets = j_block_start + j_offsets
        j_mask = (current_j_offsets < N) & (current_j_offsets >= i)
        
        # Load bb[j][i] values
        bb_indices = current_j_offsets * N + i
        bb_vals = tl.load(bb_ptr + bb_indices, mask=j_mask, other=0.0)
        
        # Calculate corresponding k values for each j
        # k starts at i * (i + 3) / 2, then k += j+1 for each iteration
        # For j from i to current_j, we need k += (i+1) + (i+2) + ... + j
        # Sum from (i+1) to j = sum from 1 to j - sum from 1 to i = j*(j+1)/2 - i*(i+1)/2
        j_vals = current_j_offsets
        k_offset = j_vals * (j_vals + 1) // 2 - i * (i + 1) // 2
        k_indices = k + k_offset
        
        # Load current flat_2d_array values
        flat_vals = tl.load(flat_2d_array_ptr + k_indices, mask=j_mask, other=0.0)
        
        # Update flat_2d_array
        flat_vals = flat_vals + bb_vals
        tl.store(flat_2d_array_ptr + k_indices, flat_vals, mask=j_mask)
        
        j_block_start += BLOCK_SIZE

def s141_triton(bb, flat_2d_array):
    N = bb.shape[0]
    
    BLOCK_SIZE = 64
    grid = (N,)
    
    s141_kernel[grid](
        bb,
        flat_2d_array,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )