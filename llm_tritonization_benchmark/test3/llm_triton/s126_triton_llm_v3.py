import torch
import triton
import triton.language as tl

@triton.jit
def s126_kernel(bb_ptr, bb_prev_ptr, flat_2d_ptr, cc_ptr, 
                i_val, LEN_2D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) + 1
    mask = j_offsets < LEN_2D
    
    # Load bb[j-1][i]
    bb_prev_offsets = (j_offsets - 1) * LEN_2D + i_val
    bb_prev_vals = tl.load(bb_prev_ptr + bb_prev_offsets, mask=mask)
    
    # Load flat_2d_array[k-1] where k = i * LEN_2D + j
    k_vals = i_val * LEN_2D + j_offsets
    flat_offsets = k_vals - 1
    flat_vals = tl.load(flat_2d_ptr + flat_offsets, mask=mask)
    
    # Load cc[j][i]
    cc_offsets = j_offsets * LEN_2D + i_val
    cc_vals = tl.load(cc_ptr + cc_offsets, mask=mask)
    
    # Compute bb[j][i] = bb[j-1][i] + flat_2d_array[k-1] * cc[j][i]
    result = bb_prev_vals + flat_vals * cc_vals
    
    # Store bb[j][i]
    bb_offsets = j_offsets * LEN_2D + i_val
    tl.store(bb_ptr + bb_offsets, result, mask=mask)

def s126_triton(bb, flat_2d_array, cc):
    LEN_2D = bb.shape[0]
    BLOCK_SIZE = 256
    
    # Create a working copy to avoid WAR dependencies
    bb_work = bb.clone()
    
    for i in range(LEN_2D):
        grid = (triton.cdiv(LEN_2D - 1, BLOCK_SIZE),)
        s126_kernel[grid](
            bb_work, bb_work, flat_2d_array, cc,
            i, LEN_2D, BLOCK_SIZE
        )
    
    bb.copy_(bb_work)