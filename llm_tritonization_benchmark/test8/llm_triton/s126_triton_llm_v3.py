import torch
import triton
import triton.language as tl

@triton.jit
def s126_kernel(bb_ptr, cc_ptr, flat_2d_array_ptr, i_val, LEN_2D, BLOCK_SIZE: tl.constexpr):
    # Get program ID and calculate offsets for j dimension
    pid = tl.program_id(0)
    j_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) + 1  # j starts from 1
    
    # Create mask for valid j indices
    j_mask = j_offsets < LEN_2D
    
    # Calculate flat_2d_array indices: k = i_val * LEN_2D + j
    k_offsets = i_val * LEN_2D + j_offsets - 1  # k-1 in original code
    
    # Load bb[j-1][i_val] (previous row, same column)
    bb_prev_offsets = (j_offsets - 1) * LEN_2D + i_val
    bb_prev_vals = tl.load(bb_ptr + bb_prev_offsets, mask=j_mask)
    
    # Load flat_2d_array[k-1]
    flat_vals = tl.load(flat_2d_array_ptr + k_offsets, mask=j_mask)
    
    # Load cc[j][i_val]
    cc_offsets = j_offsets * LEN_2D + i_val
    cc_vals = tl.load(cc_ptr + cc_offsets, mask=j_mask)
    
    # Compute bb[j][i_val] = bb[j-1][i_val] + flat_2d_array[k-1] * cc[j][i_val]
    result = bb_prev_vals + flat_vals * cc_vals
    
    # Store result back to bb[j][i_val]
    bb_curr_offsets = j_offsets * LEN_2D + i_val
    tl.store(bb_ptr + bb_curr_offsets, result, mask=j_mask)

def s126_triton(bb, cc, flat_2d_array):
    LEN_2D = bb.shape[0]
    BLOCK_SIZE = 256
    
    # Sequential loop over i dimension
    for i_val in range(LEN_2D):
        # Parallel execution over j dimension (j starts from 1)
        grid = (triton.cdiv(LEN_2D - 1, BLOCK_SIZE),)
        s126_kernel[grid](bb, cc, flat_2d_array, i_val, LEN_2D, BLOCK_SIZE)