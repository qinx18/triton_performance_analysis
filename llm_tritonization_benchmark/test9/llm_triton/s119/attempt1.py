import triton
import triton.language as tl
import torch

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, LEN_2D, i_val, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = 1 + j_offsets
    
    mask = j_idx < LEN_2D
    
    # Load bb[i_val][j] for j in [1, LEN_2D)
    bb_offsets = i_val * LEN_2D + j_idx
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=mask)
    
    # Load aa[i_val-1][j-1] for j in [1, LEN_2D)
    aa_read_offsets = (i_val - 1) * LEN_2D + (j_idx - 1)
    aa_read_vals = tl.load(aa_ptr + aa_read_offsets, mask=mask)
    
    # Compute aa[i_val][j] = aa[i_val-1][j-1] + bb[i_val][j]
    result = aa_read_vals + bb_vals
    
    # Store result to aa[i_val][j]
    aa_write_offsets = i_val * LEN_2D + j_idx
    tl.store(aa_ptr + aa_write_offsets, result, mask=mask)

def s119_triton(aa, bb):
    LEN_2D = aa.size(0)
    BLOCK_SIZE = triton.next_power_of_2(LEN_2D - 1)
    
    for i in range(1, LEN_2D):
        s119_kernel[(1,)](
            aa, bb,
            LEN_2D, i,
            BLOCK_SIZE=BLOCK_SIZE
        )