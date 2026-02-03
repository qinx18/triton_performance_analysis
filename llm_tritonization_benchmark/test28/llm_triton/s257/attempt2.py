import torch
import triton
import triton.language as tl

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, len_2d: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < len_2d
    
    for i in range(1, len_2d):
        # Load a[i-1] (scalar)
        a_prev = tl.load(a_ptr + (i - 1))
        
        # For each j in parallel, compute a[i] = aa[j][i] - a[i-1]
        # Load aa[j][i] for all j
        aa_offsets = j_offsets * len_2d + i
        aa_vals = tl.load(aa_ptr + aa_offsets, mask=j_mask)
        
        # Compute a[i] = aa[j][i] - a[i-1] for each j
        a_new = aa_vals - a_prev
        
        # Load bb[j][i] for all j
        bb_offsets = j_offsets * len_2d + i
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask)
        
        # Compute aa[j][i] = a[i] + bb[j][i] for each j
        aa_new = a_new + bb_vals
        
        # Store aa[j][i]
        tl.store(aa_ptr + aa_offsets, aa_new, mask=j_mask)
        
        # Store a[i] - use the value from the last valid j
        # Since all j compute the same value for a[i], we can use any valid one
        # Extract the first valid value
        if len_2d > 0:
            a_final = tl.load(aa_ptr + i) - tl.load(bb_ptr + i) 
            tl.store(a_ptr + i, a_final)

def s257_triton(a, aa, bb, len_2d):
    BLOCK_SIZE = triton.next_power_of_2(len_2d)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)
    
    grid = (1,)
    
    s257_kernel[grid](
        a, aa, bb, 
        len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )