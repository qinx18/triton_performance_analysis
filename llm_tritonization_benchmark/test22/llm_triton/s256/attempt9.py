import triton
import triton.language as tl
import torch

@triton.jit
def s256_kernel(a_ptr, aa_ptr, bb_ptr, d_ptr, i_val, N, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < (N - 1)
    j_indices = 1 + j_offsets
    
    # Load a[j-1] values
    a_j_minus_1 = tl.load(a_ptr + j_indices - 1, mask=j_mask)
    
    # Calculate a[j] = 1.0 - a[j-1]
    a_j = 1.0 - a_j_minus_1
    
    # Store a[j]
    tl.store(a_ptr + j_indices, a_j, mask=j_mask)
    
    # Load bb[j][i] and d[j]
    bb_offsets = j_indices * N + i_val
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask)
    d_vals = tl.load(d_ptr + j_indices, mask=j_mask)
    
    # Calculate aa[j][i] = a[j] + bb[j][i] * d[j]
    aa_vals = a_j + bb_vals * d_vals
    
    # Store aa[j][i]
    aa_offsets = j_indices * N + i_val
    tl.store(aa_ptr + aa_offsets, aa_vals, mask=j_mask)

def s256_triton(a, aa, bb, d):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    for i in range(N):
        grid = (triton.cdiv(N - 1, BLOCK_SIZE),)
        s256_kernel[grid](a, aa, bb, d, i, N, BLOCK_SIZE=BLOCK_SIZE)