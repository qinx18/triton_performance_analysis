import triton
import triton.language as tl
import torch

@triton.jit
def s2233_kernel(
    aa_ptr, aa_copy_ptr, bb_ptr, bb_copy_ptr, cc_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get the current i index (outer loop)
    i = tl.program_id(0) + 1
    
    if i >= LEN_2D:
        return
    
    # First inner loop: aa[j][i] = aa[j-1][i] + cc[j][i]
    for j_start in range(1, LEN_2D, BLOCK_SIZE):
        j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
        j_mask = (j_offsets < LEN_2D) & (j_offsets >= 1)
        
        # Current positions: aa[j][i]
        aa_curr_offsets = j_offsets * LEN_2D + i
        # Previous positions: aa[j-1][i]  
        aa_prev_offsets = (j_offsets - 1) * LEN_2D + i
        # cc positions: cc[j][i]
        cc_offsets = j_offsets * LEN_2D + i
        
        # Load from copies for reads
        aa_prev_vals = tl.load(aa_copy_ptr + aa_prev_offsets, mask=j_mask)
        cc_vals = tl.load(cc_ptr + cc_offsets, mask=j_mask)
        
        # Compute and store to original
        result = aa_prev_vals + cc_vals
        tl.store(aa_ptr + aa_curr_offsets, result, mask=j_mask)
    
    # Second inner loop: bb[i][j] = bb[i-1][j] + cc[i][j]
    for j_start in range(1, LEN_2D, BLOCK_SIZE):
        j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
        j_mask = (j_offsets < LEN_2D) & (j_offsets >= 1)
        
        # Current positions: bb[i][j]
        bb_curr_offsets = i * LEN_2D + j_offsets
        # Previous positions: bb[i-1][j]
        bb_prev_offsets = (i - 1) * LEN_2D + j_offsets
        # cc positions: cc[i][j]
        cc_offsets = i * LEN_2D + j_offsets
        
        # Load from copies for reads
        bb_prev_vals = tl.load(bb_copy_ptr + bb_prev_offsets, mask=j_mask)
        cc_vals = tl.load(cc_ptr + cc_offsets, mask=j_mask)
        
        # Compute and store to original
        result = bb_prev_vals + cc_vals
        tl.store(bb_ptr + bb_curr_offsets, result, mask=j_mask)

def s2233_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    
    # Create read-only copies to handle WAR dependencies
    aa_copy = aa.clone()
    bb_copy = bb.clone()
    
    # Launch kernel for each i from 1 to LEN_2D-1
    BLOCK_SIZE = 64
    grid = (LEN_2D - 1,)  # i ranges from 1 to LEN_2D-1
    
    s2233_kernel[grid](
        aa, aa_copy, bb, bb_copy, cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )