import triton
import triton.language as tl
import torch

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < len_2d
    
    # Load bb values for all j
    bb_vals = tl.load(bb_ptr + j_offsets, mask=j_mask, other=0.0)
    
    # Load current a[i] value
    a_val = tl.load(a_ptr)
    
    # Compute aa[j][i] = a[i] + bb[j][i] for all j
    aa_vals = a_val + bb_vals
    
    # Store aa values
    tl.store(aa_ptr + j_offsets, aa_vals, mask=j_mask)

def s257_triton(a, aa, bb):
    len_2d = aa.shape[0]
    BLOCK_SIZE = triton.next_power_of_2(len_2d)
    
    for i in range(1, len_2d):
        # Compute a[i] = aa[j][i] - a[i-1] for all j, taking last value
        for j in range(len_2d):
            a[i] = aa[j, i] - a[i-1]
        
        # Launch kernel to compute aa[j][i] = a[i] + bb[j][i] for all j
        s257_kernel[(1,)](
            a[i:i+1], aa[:, i], bb[:, i], len_2d, BLOCK_SIZE=BLOCK_SIZE
        )