import triton
import triton.language as tl
import torch

@triton.jit
def s233_kernel(
    aa_ptr, bb_ptr, cc_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Sequential processing over i dimension due to dependencies
    for i in range(1, LEN_2D):
        # First loop: aa[j][i] = aa[j-1][i] + cc[j][i]
        # This has dependency on previous j, so process sequentially
        for j in range(1, LEN_2D):
            aa_curr_offset = j * LEN_2D + i
            aa_prev_offset = (j - 1) * LEN_2D + i
            cc_offset = j * LEN_2D + i
            
            aa_prev_val = tl.load(aa_ptr + aa_prev_offset)
            cc_val = tl.load(cc_ptr + cc_offset)
            result = aa_prev_val + cc_val
            tl.store(aa_ptr + aa_curr_offset, result)
        
        # Second loop: bb[j][i] = bb[j][i-1] + cc[j][i]
        # This has dependency on previous i, so process with blocks in j
        block_start = tl.program_id(0) * BLOCK_SIZE
        j_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        j_mask = (j_offsets >= 1) & (j_offsets < LEN_2D)
        
        bb_curr_offsets = j_offsets * LEN_2D + i
        bb_prev_offsets = j_offsets * LEN_2D + (i - 1)
        cc_offsets = j_offsets * LEN_2D + i
        
        bb_prev_vals = tl.load(bb_ptr + bb_prev_offsets, mask=j_mask)
        cc_vals = tl.load(cc_ptr + cc_offsets, mask=j_mask)
        results = bb_prev_vals + cc_vals
        tl.store(bb_ptr + bb_curr_offsets, results, mask=j_mask)

def s233_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Launch kernel with single thread since we have sequential dependencies
    grid = (1,)
    
    s233_kernel[grid](
        aa, bb, cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )