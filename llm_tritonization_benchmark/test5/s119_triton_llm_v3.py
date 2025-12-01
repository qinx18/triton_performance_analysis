import torch
import triton
import triton.language as tl

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, LEN_2D: tl.constexpr, i_val: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    mask = (j_offsets >= 1) & (j_offsets < LEN_2D)
    
    # Load bb[i_val][j]
    bb_offsets = i_val * LEN_2D + j_offsets
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=mask)
    
    # Load aa[i_val-1][j-1]
    aa_read_offsets = (i_val - 1) * LEN_2D + (j_offsets - 1)
    aa_vals = tl.load(aa_ptr + aa_read_offsets, mask=mask)
    
    # Compute aa[i_val][j] = aa[i_val-1][j-1] + bb[i_val][j]
    result = aa_vals + bb_vals
    
    # Store to aa[i_val][j]
    aa_write_offsets = i_val * LEN_2D + j_offsets
    tl.store(aa_ptr + aa_write_offsets, result, mask=mask)

def s119_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    for i_val in range(1, LEN_2D):
        grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
        s119_kernel[grid](
            aa, bb, LEN_2D, i_val, BLOCK_SIZE
        )