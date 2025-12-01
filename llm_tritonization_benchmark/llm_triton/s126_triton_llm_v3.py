import triton
import triton.language as tl
import torch

@triton.jit
def s126_kernel(bb_ptr, flat_2d_array_ptr, cc_ptr, i_val, LEN_2D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = 1 + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = j_offsets < LEN_2D
    
    # Load bb[j-1][i] 
    bb_prev_offsets = (j_offsets - 1) * LEN_2D + i_val
    bb_prev = tl.load(bb_ptr + bb_prev_offsets, mask=mask)
    
    # Calculate k values: k = i_val * (LEN_2D - 1) + (j_offsets - 1) + 1
    k_vals = i_val * (LEN_2D - 1) + (j_offsets - 1) + 1
    flat_array_offsets = k_vals - 1
    flat_vals = tl.load(flat_2d_array_ptr + flat_array_offsets, mask=mask)
    
    # Load cc[j][i]
    cc_offsets = j_offsets * LEN_2D + i_val
    cc_vals = tl.load(cc_ptr + cc_offsets, mask=mask)
    
    # Compute bb[j][i] = bb[j-1][i] + flat_2d_array[k-1] * cc[j][i]
    result = bb_prev + flat_vals * cc_vals
    
    # Store bb[j][i]
    bb_offsets = j_offsets * LEN_2D + i_val
    tl.store(bb_ptr + bb_offsets, result, mask=mask)

def s126_triton(bb, flat_2d_array, cc):
    LEN_2D = bb.shape[0]
    BLOCK_SIZE = 128
    
    for i_val in range(LEN_2D):
        grid = (triton.cdiv(LEN_2D - 1, BLOCK_SIZE),)
        s126_kernel[grid](
            bb, flat_2d_array, cc,
            i_val, LEN_2D, BLOCK_SIZE
        )