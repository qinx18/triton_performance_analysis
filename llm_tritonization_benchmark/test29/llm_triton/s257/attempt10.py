import torch
import triton
import triton.language as tl

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, len_2d: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid == 0:
        for i in range(1, len_2d):
            j_offsets = tl.arange(0, BLOCK_SIZE)
            j_mask = j_offsets < len_2d
            
            aa_ptrs = aa_ptr + j_offsets * len_2d + i
            bb_ptrs = bb_ptr + j_offsets * len_2d + i
            
            aa_vals = tl.load(aa_ptrs, mask=j_mask)
            bb_vals = tl.load(bb_ptrs, mask=j_mask)
            
            a_prev = tl.load(a_ptr + (i - 1))
            
            a_new_vals = aa_vals - a_prev
            
            for idx in range(len_2d):
                if idx < BLOCK_SIZE:
                    a_new = tl.load(aa_ptr + idx * len_2d + i) - a_prev
                    tl.store(a_ptr + i, a_new)
                    aa_new = a_new + tl.load(bb_ptr + idx * len_2d + i)
                    tl.store(aa_ptr + idx * len_2d + i, aa_new)

def s257_triton(a, aa, bb, len_2d):
    BLOCK_SIZE = triton.next_power_of_2(len_2d)
    
    grid = (1,)
    s257_kernel[grid](
        a, aa, bb, len_2d, BLOCK_SIZE
    )