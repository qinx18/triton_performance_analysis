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
    
    # Compute a[i] = aa[j][i] - a[i-1] (last j wins due to overwrite pattern)
    a_i = aa_vals - a_prev
    
    # Since each j overwrites a[i], we need the last valid value
    # Find the last valid index
    valid_mask = j_offsets < LEN_2D
    last_valid_idx = LEN_2D - 1
    a_final = aa_vals[last_valid_idx] - a_prev if LEN_2D > 0 else 0.0
    
    # Use the final a[i] value for all aa[j][i] computations
    new_aa = a_final + bb_vals
    
    # Store updated aa[j][i] values
    tl.store(aa_ptr + aa_offsets, new_aa, mask=mask)
    
    # Store final a[i]
    tl.store(a_ptr + i_val, a_final)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = triton.next_power_of_2(LEN_2D)
    
    for i in range(1, LEN_2D):
        grid = (1,)
        s257_kernel[grid](a, aa, bb, i, LEN_2D, BLOCK_SIZE=BLOCK_SIZE)