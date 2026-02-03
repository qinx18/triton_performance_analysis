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
        
        # For all j, compute a[i] = aa[j][i] - a[i-1]
        # Since all j iterations write the same value to a[i], we compute it once
        # Using aa[0][i] for the computation (any j would work since result is same)
        aa_0_i = tl.load(aa_ptr + i)
        a_i = aa_0_i - a_prev
        
        # Store a[i]
        tl.store(a_ptr + i, a_i)
        
        # Now for each j, compute aa[j][i] = a[i] + bb[j][i]
        aa_offsets = j_offsets * len_2d + i
        bb_offsets = j_offsets * len_2d + i
        
        # Load bb[j][i] for all j
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask)
        
        # Compute aa[j][i] = a[i] + bb[j][i] for each j
        aa_new = a_i + bb_vals
        
        # Store aa[j][i]
        tl.store(aa_ptr + aa_offsets, aa_new, mask=j_mask)

def s257_triton(a, aa, bb, len_2d):
    BLOCK_SIZE = triton.next_power_of_2(len_2d)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)
    
    grid = (1,)
    
    s257_kernel[grid](
        a, aa, bb, 
        len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )