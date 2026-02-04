import torch
import triton
import triton.language as tl

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < len_2d
    
    for i in range(1, len_2d):
        # Load a[i-1] (scalar)
        a_prev = tl.load(a_ptr + (i - 1))
        
        # Load aa[j][i] for all j in this block
        aa_ptrs = aa_ptr + j_offsets * len_2d + i
        aa_vals = tl.load(aa_ptrs, mask=j_mask)
        
        # Load bb[j][i] for all j in this block
        bb_ptrs = bb_ptr + j_offsets * len_2d + i
        bb_vals = tl.load(bb_ptrs, mask=j_mask)
        
        # Compute a[i] = aa[j][i] - a[i-1] (last j wins due to overwrite)
        a_new_vals = aa_vals - a_prev
        
        # Store the last computed a[i] value
        if len_2d > 0:
            last_j = len_2d - 1
            if last_j < BLOCK_SIZE:
                a_new = tl.load(aa_ptr + last_j * len_2d + i) - a_prev
                tl.store(a_ptr + i, a_new)
        
        # Load the final a[i] value for aa computation
        a_final = tl.load(a_ptr + i)
        
        # Compute aa[j][i] = a[i] + bb[j][i]
        aa_new_vals = a_final + bb_vals
        
        # Store aa[j][i] for all j in this block
        tl.store(aa_ptrs, aa_new_vals, mask=j_mask)

def s257_triton(a, aa, bb, len_2d):
    BLOCK_SIZE = triton.next_power_of_2(len_2d)
    
    grid = (1,)
    s257_kernel[grid](
        a, aa, bb, len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )