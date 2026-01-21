import triton
import triton.language as tl
import torch

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, i, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    mask = j_offsets < LEN_2D
    
    # Load a[i-1] (scalar broadcast)
    a_i_minus_1 = tl.load(a_ptr + (i - 1))
    
    # Load aa[j][i] for all j
    aa_ji_offsets = j_offsets * LEN_2D + i
    aa_ji = tl.load(aa_ptr + aa_ji_offsets, mask=mask)
    
    # Load bb[j][i] for all j  
    bb_ji_offsets = j_offsets * LEN_2D + i
    bb_ji = tl.load(bb_ptr + bb_ji_offsets, mask=mask)
    
    # Compute a[i] = aa[j][i] - a[i-1] (last j wins due to overwrite)
    a_i = aa_ji - a_i_minus_1
    
    # Store final a[i] value (from last valid j)
    if LEN_2D > 0:
        final_a_i = tl.load(aa_ptr + (LEN_2D - 1) * LEN_2D + i) - a_i_minus_1
        tl.store(a_ptr + i, final_a_i)
    
    # Compute and store aa[j][i] = a[i] + bb[j][i] for all j
    new_aa_ji = a_i + bb_ji
    tl.store(aa_ptr + aa_ji_offsets, new_aa_ji, mask=mask)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    for i in range(1, LEN_2D):
        s257_kernel[grid](a, aa, bb, i, LEN_2D, BLOCK_SIZE=BLOCK_SIZE)