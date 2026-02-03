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
        
        # Process all j values for this i
        for j in range(len_2d):
            # Compute a[i] = aa[j][i] - a[i-1]
            aa_ji = tl.load(aa_ptr + j * len_2d + i)
            a_i = aa_ji - a_prev
            
            # Store a[i] (overwrites with each j)
            tl.store(a_ptr + i, a_i)
            
            # Compute aa[j][i] = a[i] + bb[j][i]
            bb_ji = tl.load(bb_ptr + j * len_2d + i)
            aa_new = a_i + bb_ji
            
            # Store aa[j][i]
            tl.store(aa_ptr + j * len_2d + i, aa_new)

def s257_triton(a, aa, bb, len_2d):
    BLOCK_SIZE = triton.next_power_of_2(len_2d)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)
    
    grid = (1,)
    
    s257_kernel[grid](
        a, aa, bb, 
        len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )