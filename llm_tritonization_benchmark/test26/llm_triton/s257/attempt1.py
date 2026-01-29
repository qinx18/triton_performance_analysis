import torch
import triton
import triton.language as tl

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, LEN_2D, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < LEN_2D
    
    # Load bb values for this i
    bb_vals = tl.load(bb_ptr + j_offsets * LEN_2D, mask=j_mask)
    
    # Load aa values for this i  
    aa_vals = tl.load(aa_ptr + j_offsets * LEN_2D, mask=j_mask)
    
    # Compute new a value (will be overwritten for each j, last j wins)
    for j in range(LEN_2D):
        if j == LEN_2D - 1:  # Only store the final value
            a_prev = tl.load(a_ptr - 1)
            aa_ji = tl.load(aa_ptr + j * LEN_2D)
            new_a = aa_ji - a_prev
            tl.store(a_ptr, new_a)
    
    # Compute and store aa values
    a_val = tl.load(a_ptr)
    new_aa_vals = a_val + bb_vals
    tl.store(aa_ptr + j_offsets * LEN_2D, new_aa_vals, mask=j_mask)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    
    BLOCK_SIZE = triton.next_power_of_2(LEN_2D)
    
    for i in range(1, LEN_2D):
        s257_kernel[(1,)](
            a[i:i+1], 
            aa[:, i:i+1], 
            bb[:, i:i+1],
            LEN_2D,
            BLOCK_SIZE=BLOCK_SIZE
        )