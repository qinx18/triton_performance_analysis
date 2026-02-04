import torch
import triton
import triton.language as tl

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, len_2d: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < len_2d
    
    for i in range(1, len_2d):
        # Sequentially process each j for this i
        for j in range(len_2d):
            # a[i] = aa[j][i] - a[i-1]
            aa_val = tl.load(aa_ptr + j * len_2d + i)
            a_prev = tl.load(a_ptr + (i - 1))
            a_new = aa_val - a_prev
            tl.store(a_ptr + i, a_new)
            
            # aa[j][i] = a[i] + bb[j][i]
            bb_val = tl.load(bb_ptr + j * len_2d + i)
            aa_new = a_new + bb_val
            tl.store(aa_ptr + j * len_2d + i, aa_new)

def s257_triton(a, aa, bb, len_2d):
    BLOCK_SIZE = triton.next_power_of_2(len_2d)
    
    grid = (1,)
    s257_kernel[grid](
        a, aa, bb, len_2d, BLOCK_SIZE
    )