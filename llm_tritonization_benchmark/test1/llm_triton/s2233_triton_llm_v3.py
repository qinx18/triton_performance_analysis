import triton
import triton.language as tl
import torch

@triton.jit
def s2233_kernel(
    aa_ptr, aa_copy_ptr, bb_ptr, bb_copy_ptr, cc_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID for i dimension (outer loop)
    i = tl.program_id(0) + 1
    
    if i >= LEN_2D:
        return
    
    # First inner loop: aa[j][i] = aa[j-1][i] + cc[j][i]
    # Process j from 1 to LEN_2D-1 in blocks
    for j_start in range(1, LEN_2D, BLOCK_SIZE):
        j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
        j_mask = j_offsets < LEN_2D
        
        # Load aa[j-1][i] from copy (read-only)
        aa_prev_offsets = (j_offsets - 1) * LEN_2D + i
        aa_prev_vals = tl.load(aa_copy_ptr + aa_prev_offsets, mask=j_mask)
        
        # Load cc[j][i]
        cc_offsets = j_offsets * LEN_2D + i
        cc_vals = tl.load(cc_ptr + cc_offsets, mask=j_mask)
        
        # Compute and store aa[j][i]
        aa_result = aa_prev_vals + cc_vals
        aa_store_offsets = j_offsets * LEN_2D + i
        tl.store(aa_ptr + aa_store_offsets, aa_result, mask=j_mask)
    
    # Second inner loop: bb[i][j] = bb[i-1][j] + cc[i][j]
    # Process j from 1 to LEN_2D-1 in blocks
    for j_start in range(1, LEN_2D, BLOCK_SIZE):
        j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
        j_mask = j_offsets < LEN_2D
        
        # Load bb[i-1][j] from copy (read-only)
        bb_prev_offsets = (i - 1) * LEN_2D + j_offsets
        bb_prev_vals = tl.load(bb_copy_ptr + bb_prev_offsets, mask=j_mask)
        
        # Load cc[i][j]
        cc_offsets = i * LEN_2D + j_offsets
        cc_vals = tl.load(cc_ptr + cc_offsets, mask=j_mask)
        
        # Compute and store bb[i][j]
        bb_result = bb_prev_vals + cc_vals
        bb_store_offsets = i * LEN_2D + j_offsets
        tl.store(bb_ptr + bb_store_offsets, bb_result, mask=j_mask)

def s2233_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Create read-only copies to handle WAR dependencies
    aa_copy = aa.clone()
    bb_copy = bb.clone()
    
    # Flatten arrays for kernel access
    aa_flat = aa.view(-1)
    aa_copy_flat = aa_copy.view(-1)
    bb_flat = bb.view(-1)
    bb_copy_flat = bb_copy.view(-1)
    cc_flat = cc.view(-1)
    
    # Launch kernel with grid over i dimension (excluding i=0)
    grid = (LEN_2D - 1,)
    
    s2233_kernel[grid](
        aa_flat, aa_copy_flat, bb_flat, bb_copy_flat, cc_flat,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )