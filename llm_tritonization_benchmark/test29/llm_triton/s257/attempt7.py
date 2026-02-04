import torch
import triton
import triton.language as tl

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, len_2d: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < len_2d
    
    for i in range(1, len_2d):
        # Load a[i-1]
        a_prev = tl.load(a_ptr + (i - 1))
        
        # Process all j values for this i
        for j in range(len_2d):
            # Compute a[i] = aa[j][i] - a[i-1]
            aa_val = tl.load(aa_ptr + j * len_2d + i)
            a_new = aa_val - a_prev
            tl.store(a_ptr + i, a_new)
            
            # Update a[i-1] for next j iteration
            a_prev = a_new
        
        # Load final a[i] value
        a_i = tl.load(a_ptr + i)
        
        # Update aa[j][i] = a[i] + bb[j][i] for all j in parallel
        aa_offsets = j_offsets * len_2d + i
        bb_vals = tl.load(bb_ptr + aa_offsets, mask=j_mask)
        new_aa_vals = a_i + bb_vals
        tl.store(aa_ptr + aa_offsets, new_aa_vals, mask=j_mask)

def s257_triton(a, aa, bb, len_2d):
    BLOCK_SIZE = triton.next_power_of_2(len_2d)
    
    grid = (1,)
    s257_kernel[grid](
        a, aa, bb, len_2d, BLOCK_SIZE
    )