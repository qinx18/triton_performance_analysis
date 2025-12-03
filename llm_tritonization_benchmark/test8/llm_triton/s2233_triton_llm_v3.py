import torch
import triton
import triton.language as tl

@triton.jit
def s2233_kernel(
    aa_ptr, bb_ptr, cc_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Sequential execution over i dimension due to dependencies
    for i in range(1, LEN_2D):
        # First inner loop: aa[j][i] = aa[j-1][i] + cc[j][i]
        # This has WAR dependency on aa, must be sequential in j
        for j in range(1, LEN_2D):
            aa_curr_offset = j * LEN_2D + i
            aa_prev_offset = (j - 1) * LEN_2D + i
            cc_offset = j * LEN_2D + i
            
            aa_prev_val = tl.load(aa_ptr + aa_prev_offset)
            cc_val = tl.load(cc_ptr + cc_offset)
            result = aa_prev_val + cc_val
            tl.store(aa_ptr + aa_curr_offset, result)
        
        # Second inner loop: bb[i][j] = bb[i-1][j] + cc[i][j]  
        # This has WAR dependency on bb, must be sequential in j
        for j in range(1, LEN_2D):
            bb_curr_offset = i * LEN_2D + j
            bb_prev_offset = (i - 1) * LEN_2D + j
            cc_offset = i * LEN_2D + j
            
            bb_prev_val = tl.load(bb_ptr + bb_prev_offset)
            cc_val = tl.load(cc_ptr + cc_offset)
            result = bb_prev_val + cc_val
            tl.store(bb_ptr + bb_curr_offset, result)

def s2233_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Launch single thread block since all loops must be sequential
    grid = (1,)
    
    s2233_kernel[grid](
        aa, bb, cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE,
    )