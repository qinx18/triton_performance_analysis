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
        
        for j in range(len_2d):
            # Load aa[j][i]
            aa_val = tl.load(aa_ptr + j * len_2d + i)
            
            # Compute a[i] = aa[j][i] - a[i-1]
            a_new = aa_val - a_prev
            
            # Store a[i]
            tl.store(a_ptr + i, a_new)
            
            # Load bb[j][i]
            bb_val = tl.load(bb_ptr + j * len_2d + i)
            
            # Compute aa[j][i] = a[i] + bb[j][i]
            aa_new = a_new + bb_val
            
            # Store aa[j][i]
            tl.store(aa_ptr + j * len_2d + i, aa_new)

def s257_triton(a, aa, bb, len_2d):
    BLOCK_SIZE = triton.next_power_of_2(len_2d)
    
    grid = (1,)
    s257_kernel[grid](
        a, aa, bb, len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )