import torch
import triton
import triton.language as tl

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, i_val, LEN_2D, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < LEN_2D
    
    # Load a[i-1]
    a_i_minus_1 = tl.load(a_ptr + (i_val - 1))
    
    # Load aa[j][i] for all j values
    aa_indices = j_offsets * LEN_2D + i_val
    aa_vals = tl.load(aa_ptr + aa_indices, mask=j_mask)
    
    # Load bb[j][i] for all j values
    bb_indices = j_offsets * LEN_2D + i_val
    bb_vals = tl.load(bb_ptr + bb_indices, mask=j_mask)
    
    # Compute a[i] = aa[j][i] - a[i-1]
    # Since this overwrites for each j, we use the last valid j value
    a_vals = aa_vals - a_i_minus_1
    
    # Store the last valid a[i] value (overwrite pattern)
    if LEN_2D > 0:
        final_a_val = tl.load(aa_ptr + (LEN_2D - 1) * LEN_2D + i_val) - a_i_minus_1
        tl.store(a_ptr + i_val, final_a_val)
    
    # Compute and store aa[j][i] = a[i] + bb[j][i]
    new_aa_vals = a_vals + bb_vals
    tl.store(aa_ptr + aa_indices, new_aa_vals, mask=j_mask)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    
    BLOCK_SIZE = triton.next_power_of_2(LEN_2D)
    
    for i in range(1, LEN_2D):
        s257_kernel[(1,)](
            a, aa, bb, i, LEN_2D, BLOCK_SIZE
        )