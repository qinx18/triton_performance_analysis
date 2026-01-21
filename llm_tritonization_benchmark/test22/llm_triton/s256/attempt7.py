import triton
import triton.language as tl
import torch

@triton.jit
def s256_kernel(a_ptr, aa_ptr, bb_ptr, d_ptr, j_val, N, BLOCK_SIZE: tl.constexpr):
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_mask = i_offsets < N
    
    # Calculate a[j] = 1.0 - a[j-1] (scalar operation)
    a_j_minus_1 = tl.load(a_ptr + j_val - 1)
    a_j = 1.0 - a_j_minus_1
    tl.store(a_ptr + j_val, a_j)
    
    # Load bb[j][i] and d[j] for all i
    bb_offsets = j_val * N + i_offsets
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=i_mask)
    d_j = tl.load(d_ptr + j_val)
    
    # Calculate aa[j][i] = a[j] + bb[j][i] * d[j]
    aa_vals = a_j + bb_vals * d_j
    
    # Store aa[j][i]
    aa_offsets = j_val * N + i_offsets
    tl.store(aa_ptr + aa_offsets, aa_vals, mask=i_mask)

def s256_triton(a, aa, bb, d):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Sequential loop over j, parallel over i
    for j in range(1, N):
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        s256_kernel[grid](a, aa, bb, d, j, N, BLOCK_SIZE=BLOCK_SIZE)