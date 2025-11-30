import triton
import triton.language as tl
import torch

@triton.jit
def s126_kernel(bb_ptr, bb_prev_ptr, flat_2d_array_ptr, cc_ptr, 
                i_val, LEN_2D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) + 1
    mask = j_offsets < LEN_2D
    
    # Calculate k values for each j
    k_vals = i_val * LEN_2D + j_offsets - 1
    
    # Load bb[j-1][i] values
    bb_prev_offsets = (j_offsets - 1) * LEN_2D + i_val
    bb_prev_vals = tl.load(bb_prev_ptr + bb_prev_offsets, mask=mask)
    
    # Load flat_2d_array[k-1] values
    flat_vals = tl.load(flat_2d_array_ptr + k_vals, mask=mask)
    
    # Load cc[j][i] values
    cc_offsets = j_offsets * LEN_2D + i_val
    cc_vals = tl.load(cc_ptr + cc_offsets, mask=mask)
    
    # Compute bb[j][i] = bb[j-1][i] + flat_2d_array[k-1] * cc[j][i]
    result = bb_prev_vals + flat_vals * cc_vals
    
    # Store results
    bb_offsets = j_offsets * LEN_2D + i_val
    tl.store(bb_ptr + bb_offsets, result, mask=mask)

def s126_triton(a, b, c, d, e, aa, bb, cc, flat_2d_array):
    LEN_2D = bb.shape[0]
    BLOCK_SIZE = 256
    
    # Sequential loop over i
    for i_val in range(LEN_2D):
        grid = (triton.cdiv(LEN_2D - 1, BLOCK_SIZE),)
        s126_kernel[grid](
            bb, bb, flat_2d_array, cc,
            i_val, LEN_2D, BLOCK_SIZE
        )