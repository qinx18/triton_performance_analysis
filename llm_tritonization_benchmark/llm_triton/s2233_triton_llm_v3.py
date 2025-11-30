import triton
import triton.language as tl
import torch

@triton.jit
def s2233_kernel(
    aa_ptr, aa_copy_ptr, bb_ptr, bb_copy_ptr, cc_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get the current i index
    i = tl.program_id(0) + 1
    
    if i >= LEN_2D:
        return
    
    # Process j indices in blocks
    j_block_start = tl.program_id(1) * BLOCK_SIZE + 1
    j_offsets = j_block_start + tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < LEN_2D
    
    # First loop: aa[j][i] = aa[j-1][i] + cc[j][i]
    if j_block_start >= 1:  # Only process if we have valid j-1 indices
        # Load aa[j-1][i] values
        aa_prev_offsets = (j_offsets - 1) * LEN_2D + i
        aa_prev_vals = tl.load(aa_copy_ptr + aa_prev_offsets, mask=j_mask)
        
        # Load cc[j][i] values
        cc_offsets = j_offsets * LEN_2D + i
        cc_vals = tl.load(cc_ptr + cc_offsets, mask=j_mask)
        
        # Compute and store aa[j][i] = aa[j-1][i] + cc[j][i]
        aa_result = aa_prev_vals + cc_vals
        aa_write_offsets = j_offsets * LEN_2D + i
        tl.store(aa_ptr + aa_write_offsets, aa_result, mask=j_mask)
    
    # Second loop: bb[i][j] = bb[i-1][j] + cc[i][j]
    if i >= 1:  # Only process if we have valid i-1 index
        # Load bb[i-1][j] values
        bb_prev_offsets = (i - 1) * LEN_2D + j_offsets
        bb_prev_vals = tl.load(bb_copy_ptr + bb_prev_offsets, mask=j_mask)
        
        # Load cc[i][j] values
        cc_i_offsets = i * LEN_2D + j_offsets
        cc_i_vals = tl.load(cc_ptr + cc_i_offsets, mask=j_mask)
        
        # Compute and store bb[i][j] = bb[i-1][j] + cc[i][j]
        bb_result = bb_prev_vals + cc_i_vals
        bb_write_offsets = i * LEN_2D + j_offsets
        tl.store(bb_ptr + bb_write_offsets, bb_result, mask=j_mask)

def s2233_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    
    # Create read-only copies to handle WAR dependencies
    aa_copy = aa.clone()
    bb_copy = bb.clone()
    
    # Flatten arrays for kernel access
    aa_flat = aa.view(-1)
    aa_copy_flat = aa_copy.view(-1)
    bb_flat = bb.view(-1)
    bb_copy_flat = bb_copy.view(-1)
    cc_flat = cc.view(-1)
    
    BLOCK_SIZE = 64
    
    # Grid dimensions: (i_blocks, j_blocks)
    grid = (LEN_2D - 1, triton.cdiv(LEN_2D - 1, BLOCK_SIZE))
    
    s2233_kernel[grid](
        aa_flat, aa_copy_flat, bb_flat, bb_copy_flat, cc_flat,
        LEN_2D, BLOCK_SIZE
    )