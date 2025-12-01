import torch
import triton
import triton.language as tl

@triton.jit
def s1119_kernel(
    aa_ptr,
    bb_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    i_val: tl.constexpr,
):
    pid = tl.program_id(0)
    j_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < LEN_2D
    
    # Calculate memory offsets for aa[i][j], aa[i-1][j], and bb[i][j]
    aa_curr_offsets = i_val * LEN_2D + j_offsets
    aa_prev_offsets = (i_val - 1) * LEN_2D + j_offsets
    bb_offsets = i_val * LEN_2D + j_offsets
    
    # Vectorized loads
    aa_prev_vals = tl.load(aa_ptr + aa_prev_offsets, mask=j_mask)
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask)
    
    # Vectorized computation
    result = aa_prev_vals + bb_vals
    
    # Vectorized store
    tl.store(aa_ptr + aa_curr_offsets, result, mask=j_mask)

def s1119_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Sequential loop over i, parallel over j
    for i_val in range(1, LEN_2D):
        grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
        s1119_kernel[grid](
            aa,
            bb,
            LEN_2D,
            BLOCK_SIZE,
            i_val,
        )