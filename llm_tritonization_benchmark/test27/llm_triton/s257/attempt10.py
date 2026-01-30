import triton
import triton.language as tl
import torch

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, a_prev, len_2d, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < len_2d
    
    # Load aa[j][i] and bb[j][i] values for all j
    aa_vals = tl.load(aa_ptr + j_offsets, mask=j_mask, other=0.0)
    bb_vals = tl.load(bb_ptr + j_offsets, mask=j_mask, other=0.0)
    
    # Compute a[i] = aa[j][i] - a[i-1] for all j (last j wins)
    a_vals = aa_vals - a_prev
    
    # Store the last a[i] value (from the last valid j)
    if len_2d > 0:
        last_j = len_2d - 1
        a_final = aa_vals[last_j] - a_prev if last_j < BLOCK_SIZE else tl.load(aa_ptr + last_j) - a_prev
        tl.store(a_ptr, a_final)
    
    # Compute aa[j][i] = a[i] + bb[j][i] for all j
    aa_new_vals = a_vals + bb_vals
    
    # Store aa values
    tl.store(aa_ptr + j_offsets, aa_new_vals, mask=j_mask)

def s257_triton(a, aa, bb):
    len_2d = aa.shape[0]
    BLOCK_SIZE = triton.next_power_of_2(len_2d)
    
    for i in range(1, len_2d):
        # Launch kernel to compute both operations
        s257_kernel[(1,)](
            a[i:i+1], aa[:, i], bb[:, i], a[i-1], len_2d, BLOCK_SIZE=BLOCK_SIZE
        )