import torch
import triton
import triton.language as tl

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < len_2d
    
    for i in range(1, len_2d):
        # Load a[i-1]
        a_prev = tl.load(a_ptr + (i - 1))
        
        # Load aa[j][i] for all j
        aa_vals = tl.load(aa_ptr + j_offsets * len_2d + i, mask=j_mask)
        
        # Compute a[i] = aa[j][i] - a[i-1] (using last j value)
        a_new = tl.where(j_mask, aa_vals - a_prev, 0.0)
        a_final = tl.max(tl.where(j_mask, a_new, float('-inf')))
        
        # Store a[i]
        tl.store(a_ptr + i, a_final)
        
        # Load bb[j][i] for all j
        bb_vals = tl.load(bb_ptr + j_offsets * len_2d + i, mask=j_mask)
        
        # Compute aa[j][i] = a[i] + bb[j][i]
        aa_new = a_final + bb_vals
        
        # Store aa[j][i] for all j
        tl.store(aa_ptr + j_offsets * len_2d + i, aa_new, mask=j_mask)

def s257_triton(a, aa, bb, len_2d):
    BLOCK_SIZE = triton.next_power_of_2(len_2d)
    
    grid = (1,)
    s257_kernel[grid](
        a, aa, bb, len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )