import torch
import triton
import triton.language as tl

@triton.jit
def s1115_kernel(aa_ptr, bb_ptr, cc_ptr, i_val, LEN_2D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = j_offsets < LEN_2D
    
    aa_row_ptr = aa_ptr + i_val * LEN_2D
    bb_row_ptr = bb_ptr + i_val * LEN_2D
    
    aa_vals = tl.load(aa_row_ptr + j_offsets, mask=mask)
    bb_vals = tl.load(bb_row_ptr + j_offsets, mask=mask)
    
    cc_col_ptrs = cc_ptr + j_offsets * LEN_2D + i_val
    cc_vals = tl.load(cc_col_ptrs, mask=mask)
    
    result = aa_vals * cc_vals + bb_vals
    
    tl.store(aa_row_ptr + j_offsets, result, mask=mask)

def s1115_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    for i_val in range(LEN_2D):
        grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
        s1115_kernel[grid](
            aa, bb, cc, i_val, LEN_2D, BLOCK_SIZE
        )