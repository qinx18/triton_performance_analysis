import triton
import triton.language as tl
import torch

@triton.jit
def s141_kernel(bb_ptr, flat_2d_array_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Get program ID for the i dimension
    pid = tl.program_id(0)
    i = pid
    
    if i >= N:
        return
    
    # Calculate initial k value: k = (i+1) * ((i+1) - 1) / 2 + (i+1) - 1
    # This simplifies to: k = i * (i + 1) / 2 + i = i * (i + 3) / 2
    k_start = (i + 1) * i // 2 + i
    
    # Process the inner loop in blocks
    j_start = i
    j_end = N
    
    for j_block_start in range(j_start, j_end, BLOCK_SIZE):
        j_block_end = min(j_block_start + BLOCK_SIZE, j_end)
        block_size = j_block_end - j_block_start
        
        # Create offsets for this block
        j_offsets = tl.arange(0, BLOCK_SIZE)
        j_indices = j_block_start + j_offsets
        j_mask = j_indices < j_end
        
        # Calculate k values for this block
        # k starts at k_start + sum of (j+1) for j in [i, j_block_start)
        k_offset = 0
        for j in range(j_start, j_block_start):
            k_offset += j + 1
        
        # Load bb values: bb[j][i]
        bb_offsets = j_indices * N + i
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask, other=0.0)
        
        # For each j in the block, update flat_2d_array[k]
        for idx in range(block_size):
            if j_block_start + idx < j_end:
                j = j_block_start + idx
                k = k_start + k_offset
                
                # Load current value, add bb value, store back
                current_val = tl.load(flat_2d_array_ptr + k)
                bb_val = tl.load(bb_ptr + j * N + i)
                new_val = current_val + bb_val
                tl.store(flat_2d_array_ptr + k, new_val)
                
                # Update k for next iteration: k += j+1
                k_offset += j + 1

def s141_triton(bb, flat_2d_array):
    N = bb.shape[0]
    
    # Launch kernel with one thread per i value
    BLOCK_SIZE = 32
    grid = (N,)
    
    s141_kernel[grid](
        bb, flat_2d_array, N, BLOCK_SIZE
    )
    
    return flat_2d_array