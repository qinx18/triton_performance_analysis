import triton
import triton.language as tl
import torch

@triton.jit
def s256_kernel(a_ptr, a_copy_ptr, aa_ptr, bb_ptr, d_ptr, j_val, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_mask = i_offsets < LEN_2D
    
    # Load a[j-1] from copy (scalar broadcast)
    a_j_minus_1 = tl.load(a_copy_ptr + (j_val - 1))
    
    # Compute a[j] = 1.0 - a[j-1]
    a_j = 1.0 - a_j_minus_1
    
    # Store a[j] (scalar)
    tl.store(a_ptr + j_val, a_j)
    
    # Load bb[j][i] and d[j] for vectorized computation
    bb_j_i = tl.load(bb_ptr + j_val * LEN_2D + i_offsets, mask=i_mask)
    d_j = tl.load(d_ptr + j_val)
    
    # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
    aa_j_i = a_j + bb_j_i * d_j
    
    # Store aa[j][i]
    tl.store(aa_ptr + j_val * LEN_2D + i_offsets, aa_j_i, mask=i_mask)

def s256_triton(a, aa, bb, d):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = triton.next_power_of_2(LEN_2D)
    
    # Create read-only copy of a
    a_copy = a.clone()
    
    # Sequential loop over j, parallel over i
    for j in range(1, LEN_2D):
        grid = (1,)
        s256_kernel[grid](
            a, a_copy, aa, bb, d, j, LEN_2D, BLOCK_SIZE
        )