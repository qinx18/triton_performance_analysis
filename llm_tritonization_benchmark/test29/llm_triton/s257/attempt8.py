import torch
import triton
import triton.language as tl

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, len_2d: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < len_2d
    
    for i in range(1, len_2d):
        # Start with a[i-1]
        a_val = tl.load(a_ptr + (i - 1))
        
        # Process all j values sequentially for this i
        for j in range(len_2d):
            # a[i] = aa[j][i] - a[i-1]
            aa_val = tl.load(aa_ptr + j * len_2d + i)
            a_val = aa_val - a_val
            
        # Store final a[i] value
        tl.store(a_ptr + i, a_val)
        
        # Update aa[j][i] = a[i] + bb[j][i] for all j in parallel
        aa_offsets = j_offsets * len_2d + i
        bb_vals = tl.load(bb_ptr + aa_offsets, mask=j_mask)
        new_aa_vals = a_val + bb_vals
        tl.store(aa_ptr + aa_offsets, new_aa_vals, mask=j_mask)

def s257_triton(a, aa, bb, len_2d):
    BLOCK_SIZE = triton.next_power_of_2(len_2d)
    
    grid = (1,)
    s257_kernel[grid](
        a, aa, bb, len_2d, BLOCK_SIZE
    )