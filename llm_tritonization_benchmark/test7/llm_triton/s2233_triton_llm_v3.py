import torch
import triton
import triton.language as tl

@triton.jit
def s2233_kernel(
    aa_ptr,
    bb_ptr,
    cc_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID for i dimension
    pid = tl.program_id(axis=0)
    i = pid + 1  # Start from 1
    
    if i >= LEN_2D:
        return
    
    # Define offsets once at the start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # First loop: aa[j][i] = aa[j-1][i] + cc[j][i]
    # This has a j-1 dependency, so must be done sequentially
    for j_start in range(1, LEN_2D, BLOCK_SIZE):
        j_end = min(j_start + BLOCK_SIZE, LEN_2D)
        actual_block = j_end - j_start
        
        j_offsets = j_start + offsets
        mask = offsets < actual_block
        
        # Load aa[j-1][i]
        prev_offsets = (j_offsets - 1) * LEN_2D + i
        aa_prev = tl.load(aa_ptr + prev_offsets, mask=mask)
        
        # Load cc[j][i]
        cc_offsets = j_offsets * LEN_2D + i
        cc_vals = tl.load(cc_ptr + cc_offsets, mask=mask)
        
        # Compute and store aa[j][i]
        result = aa_prev + cc_vals
        aa_offsets = j_offsets * LEN_2D + i
        tl.store(aa_ptr + aa_offsets, result, mask=mask)
    
    # Second loop: bb[i][j] = bb[i-1][j] + cc[i][j]
    # Load bb[i-1][j] once for all j values
    for j_start in range(1, LEN_2D, BLOCK_SIZE):
        j_end = min(j_start + BLOCK_SIZE, LEN_2D)
        actual_block = j_end - j_start
        
        j_offsets = j_start + offsets
        mask = offsets < actual_block
        
        # Load bb[i-1][j]
        bb_prev_offsets = (i - 1) * LEN_2D + j_offsets
        bb_prev = tl.load(bb_ptr + bb_prev_offsets, mask=mask)
        
        # Load cc[i][j]
        cc_offsets = i * LEN_2D + j_offsets
        cc_vals = tl.load(cc_ptr + cc_offsets, mask=mask)
        
        # Compute and store bb[i][j]
        result = bb_prev + cc_vals
        bb_offsets = i * LEN_2D + j_offsets
        tl.store(bb_ptr + bb_offsets, result, mask=mask)

def s2233_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 32
    
    # Launch kernel for each i from 1 to LEN_2D-1
    grid = (LEN_2D - 1,)
    
    s2233_kernel[grid](
        aa,
        bb,
        cc,
        LEN_2D,
        BLOCK_SIZE,
    )