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
    
    # The j loop overwrites a[i] each time, so we need the result from j=LEN_2D-1
    # But we need to compute aa[j][i] using the a[i] from that same j iteration
    
    # For each j: a[i] = aa[j][i] - a[i-1], then aa[j][i] = a[i] + bb[j][i]
    a_new_vals = aa_vals - a_prev
    aa_new_vals = a_new_vals + bb_vals
    
    # Store updated aa[j][i] values
    tl.store(aa_ptr + aa_offsets, aa_new_vals, mask=mask)
    
    # The final a[i] value is from the last j iteration (j=LEN_2D-1)
    if LEN_2D > 0:
        final_j = LEN_2D - 1
        final_aa_val = tl.load(aa_ptr + final_j * LEN_2D + i_val)
        final_a_val = final_aa_val - bb_vals[final_j]
        tl.store(a_ptr + i_val, final_a_val)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = triton.next_power_of_2(LEN_2D)
    
    for i in range(1, LEN_2D):
        grid = (1,)
        s257_kernel[grid](a, aa, bb, i, LEN_2D, BLOCK_SIZE=BLOCK_SIZE)