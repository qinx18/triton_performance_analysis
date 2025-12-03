import torch
import triton
import triton.language as tl

@triton.jit
def s126_kernel(bb_ptr, cc_ptr, flat_2d_array_ptr, i, LEN_2D, BLOCK_SIZE: tl.constexpr):
    # Get program ID for j dimension
    pid = tl.program_id(0)
    
    # Calculate j offsets starting from 1
    j_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) + 1
    j_mask = j_offsets < LEN_2D
    
    # Calculate k for current i and j values
    # k starts at 1 and increments by 1 for each (i,j) pair in row-major order
    # For position (i,j): k = i * LEN_2D + j
    k_values = i * LEN_2D + j_offsets
    k_offsets = k_values - 1  # k-1 for indexing
    
    # Load bb[j-1][i] values
    bb_prev_offsets = (j_offsets - 1) * LEN_2D + i
    bb_prev_mask = j_mask & (j_offsets > 0)
    bb_prev = tl.load(bb_ptr + bb_prev_offsets, mask=bb_prev_mask, other=0.0)
    
    # Load flat_2d_array[k-1] values
    flat_mask = j_mask & (k_offsets >= 0) & (k_offsets < LEN_2D * LEN_2D)
    flat_vals = tl.load(flat_2d_array_ptr + k_offsets, mask=flat_mask, other=0.0)
    
    # Load cc[j][i] values
    cc_offsets = j_offsets * LEN_2D + i
    cc_vals = tl.load(cc_ptr + cc_offsets, mask=j_mask, other=0.0)
    
    # Compute bb[j][i] = bb[j-1][i] + flat_2d_array[k-1] * cc[j][i]
    result = bb_prev + flat_vals * cc_vals
    
    # Store result to bb[j][i]
    bb_offsets = j_offsets * LEN_2D + i
    tl.store(bb_ptr + bb_offsets, result, mask=j_mask)

def s126_triton(bb, cc, flat_2d_array):
    LEN_2D = bb.shape[0]
    
    # Block size for j dimension (excluding j=0)
    BLOCK_SIZE = 64
    
    # Sequential loop over i
    for i in range(LEN_2D):
        # Parallel processing of j values from 1 to LEN_2D-1
        grid_size = triton.cdiv(LEN_2D - 1, BLOCK_SIZE)
        
        s126_kernel[(grid_size,)](
            bb, cc, flat_2d_array,
            i, LEN_2D,
            BLOCK_SIZE=BLOCK_SIZE
        )