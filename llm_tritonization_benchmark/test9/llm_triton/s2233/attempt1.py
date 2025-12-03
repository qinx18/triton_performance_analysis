import torch
import triton
import triton.language as tl

@triton.jit
def s2233_kernel(
    aa_ptr, bb_ptr, cc_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    i = tl.program_id(0) + 1
    
    if i >= LEN_2D:
        return
    
    # First loop: aa[j][i] = aa[j-1][i] + cc[j][i]
    for j_start in range(1, LEN_2D, BLOCK_SIZE):
        j_offsets = tl.arange(0, BLOCK_SIZE)
        j_indices = j_start + j_offsets
        j_mask = j_indices < LEN_2D
        
        # Load aa[j-1][i]
        aa_prev_offsets = (j_indices - 1) * LEN_2D + i
        aa_prev_vals = tl.load(aa_ptr + aa_prev_offsets, mask=j_mask, other=0.0)
        
        # Load cc[j][i]
        cc_offsets = j_indices * LEN_2D + i
        cc_vals = tl.load(cc_ptr + cc_offsets, mask=j_mask, other=0.0)
        
        # Compute and store aa[j][i]
        result = aa_prev_vals + cc_vals
        aa_offsets = j_indices * LEN_2D + i
        tl.store(aa_ptr + aa_offsets, result, mask=j_mask)
    
    # Second loop: bb[i][j] = bb[i-1][j] + cc[i][j]
    for j_start in range(1, LEN_2D, BLOCK_SIZE):
        j_offsets = tl.arange(0, BLOCK_SIZE)
        j_indices = j_start + j_offsets
        j_mask = j_indices < LEN_2D
        
        # Load bb[i-1][j]
        bb_prev_offsets = (i - 1) * LEN_2D + j_indices
        bb_prev_vals = tl.load(bb_ptr + bb_prev_offsets, mask=j_mask, other=0.0)
        
        # Load cc[i][j]
        cc_offsets = i * LEN_2D + j_indices
        cc_vals = tl.load(cc_ptr + cc_offsets, mask=j_mask, other=0.0)
        
        # Compute and store bb[i][j]
        result = bb_prev_vals + cc_vals
        bb_offsets = i * LEN_2D + j_indices
        tl.store(bb_ptr + bb_offsets, result, mask=j_mask)

def s2233_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = min(128, triton.next_power_of_2(LEN_2D))
    
    grid = (LEN_2D - 1,)
    
    s2233_kernel[grid](
        aa, bb, cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )