import torch
import triton
import triton.language as tl

@triton.jit
def s2233_kernel(
    aa_ptr, aa_copy_ptr, bb_ptr, bb_copy_ptr, cc_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID for i dimension
    i = tl.program_id(0) + 1
    
    if i >= LEN_2D:
        return
    
    # First loop: aa[j][i] = aa[j-1][i] + cc[j][i]
    for j_start in range(1, LEN_2D, BLOCK_SIZE):
        j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
        mask = (j_offsets < LEN_2D)
        
        # Load from read-only copies
        aa_prev_offsets = (j_offsets - 1) * LEN_2D + i
        aa_prev_vals = tl.load(aa_copy_ptr + aa_prev_offsets, mask=mask)
        
        cc_offsets = j_offsets * LEN_2D + i
        cc_vals = tl.load(cc_ptr + cc_offsets, mask=mask)
        
        # Compute and store to original
        result = aa_prev_vals + cc_vals
        aa_offsets = j_offsets * LEN_2D + i
        tl.store(aa_ptr + aa_offsets, result, mask=mask)
    
    # Second loop: bb[i][j] = bb[i-1][j] + cc[i][j]
    for j_start in range(1, LEN_2D, BLOCK_SIZE):
        j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
        mask = (j_offsets < LEN_2D)
        
        # Load from read-only copies
        bb_prev_offsets = (i - 1) * LEN_2D + j_offsets
        bb_prev_vals = tl.load(bb_copy_ptr + bb_prev_offsets, mask=mask)
        
        cc_offsets = i * LEN_2D + j_offsets
        cc_vals = tl.load(cc_ptr + cc_offsets, mask=mask)
        
        # Compute and store to original
        result = bb_prev_vals + cc_vals
        bb_offsets = i * LEN_2D + j_offsets
        tl.store(bb_ptr + bb_offsets, result, mask=mask)

def s2233_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    
    # Create read-only copies to handle WAR dependencies
    aa_copy = aa.clone()
    bb_copy = bb.clone()
    
    BLOCK_SIZE = 64
    grid = (LEN_2D - 1,)
    
    s2233_kernel[grid](
        aa, aa_copy, bb, bb_copy, cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )