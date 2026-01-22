import torch
import triton
import triton.language as tl

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, i_idx, len_2d, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < len_2d
    
    # Load a[i] and a[i-1]
    a_i_minus_1 = tl.load(a_ptr + i_idx - 1)
    
    # Load aa[j][i] for all j values
    aa_offsets = j_offsets * len_2d + i_idx
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=j_mask)
    
    # Load bb[j][i] for all j values
    bb_offsets = j_offsets * len_2d + i_idx
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask)
    
    # Compute a[i] = aa[j][i] - a[i-1] for all j (last j wins due to overwrite)
    a_i_vals = aa_vals - a_i_minus_1
    
    # Store the final a[i] value (from last j iteration)
    if len_2d > 0:
        final_a_i = tl.load(aa_ptr + (len_2d - 1) * len_2d + i_idx) - a_i_minus_1
        tl.store(a_ptr + i_idx, final_a_i)
    
    # Compute and store aa[j][i] = a[i] + bb[j][i] for all j
    new_aa_vals = a_i_vals + bb_vals
    tl.store(aa_ptr + aa_offsets, new_aa_vals, mask=j_mask)

def s257_triton(a, aa, bb):
    len_2d = aa.shape[0]
    
    BLOCK_SIZE = 256
    
    # Sequential loop over i from 1 to len_2d-1
    for i in range(1, len_2d):
        grid = (triton.cdiv(len_2d, BLOCK_SIZE),)
        s257_kernel[grid](
            a, aa, bb, i, len_2d,
            BLOCK_SIZE=BLOCK_SIZE
        )