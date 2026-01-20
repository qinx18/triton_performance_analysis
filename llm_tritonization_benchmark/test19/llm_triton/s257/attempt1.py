import torch
import triton
import triton.language as tl

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, i_val, LEN_2D, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    mask = j_offsets < LEN_2D
    
    # Load a[i-1] (scalar broadcast)
    a_prev = tl.load(a_ptr + (i_val - 1))
    
    # Load aa[j][i] for all j values
    aa_offsets = j_offsets * LEN_2D + i_val
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=mask)
    
    # Load bb[j][i] for all j values
    bb_offsets = j_offsets * LEN_2D + i_val
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=mask)
    
    # Compute a[i] = aa[j][i] - a[i-1]
    # Since all j iterations overwrite a[i], we take the last valid j value
    a_new = aa_vals - a_prev
    
    # Store the final a[i] value (from last j iteration)
    if LEN_2D > 0:
        final_a = tl.load(aa_ptr + (LEN_2D - 1) * LEN_2D + i_val) - a_prev
        tl.store(a_ptr + i_val, final_a)
    
    # Compute and store aa[j][i] = a[i] + bb[j][i] for all j
    aa_new = a_new + bb_vals
    tl.store(aa_ptr + aa_offsets, aa_new, mask=mask)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = triton.next_power_of_2(LEN_2D)
    
    for i in range(1, LEN_2D):
        grid = (1,)
        s257_kernel[grid](a, aa, bb, i, LEN_2D, BLOCK_SIZE=BLOCK_SIZE)