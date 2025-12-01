import torch
import triton
import triton.language as tl

@triton.jit
def s1119_kernel(aa_ptr, bb_ptr, LEN_2D: tl.constexpr, i_val: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    j_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = j_offsets < LEN_2D
    
    # Calculate memory offsets for aa[i][j], aa[i-1][j], and bb[i][j]
    aa_i_offset = i_val * LEN_2D + j_offsets
    aa_i_minus_1_offset = (i_val - 1) * LEN_2D + j_offsets
    bb_i_offset = i_val * LEN_2D + j_offsets
    
    # Load aa[i-1][j] and bb[i][j]
    aa_prev = tl.load(aa_ptr + aa_i_minus_1_offset, mask=mask)
    bb_curr = tl.load(bb_ptr + bb_i_offset, mask=mask)
    
    # Compute aa[i][j] = aa[i-1][j] + bb[i][j]
    result = aa_prev + bb_curr
    
    # Store result
    tl.store(aa_ptr + aa_i_offset, result, mask=mask)

def s1119_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 128
    
    for i in range(1, LEN_2D):
        grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
        s1119_kernel[grid](
            aa, bb, 
            LEN_2D, i, BLOCK_SIZE
        )