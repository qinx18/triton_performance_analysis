import triton
import triton.language as tl
import torch

@triton.jit
def s256_kernel(a_ptr, aa_ptr, bb_ptr, d_ptr, i_val, N, BLOCK_SIZE: tl.constexpr):
    # Get block of j indices
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_indices = 1 + j_offsets
    j_mask = j_indices < N
    
    # Load a[j-1] values
    a_prev_offsets = j_indices - 1
    a_prev_mask = a_prev_offsets < N
    a_prev = tl.load(a_ptr + a_prev_offsets, mask=a_prev_mask, other=0.0)
    
    # Compute a[j] = 1.0 - a[j-1]
    a_new = 1.0 - a_prev
    
    # Store a[j] values
    tl.store(a_ptr + j_indices, a_new, mask=j_mask)
    
    # Load bb[j][i] and d[j] values
    bb_offsets = j_indices * N + i_val
    bb_mask = j_mask & (i_val < N)
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=bb_mask, other=0.0)
    
    d_vals = tl.load(d_ptr + j_indices, mask=j_mask, other=0.0)
    
    # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
    aa_new = a_new + bb_vals * d_vals
    
    # Store aa[j][i] values
    aa_offsets = j_indices * N + i_val
    tl.store(aa_ptr + aa_offsets, aa_new, mask=bb_mask)

def s256_triton(a, aa, bb, d):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Sequential loop over i, parallel processing of j
    for i in range(N):
        # Launch kernel for this i value
        grid = (triton.cdiv(N-1, BLOCK_SIZE),)
        s256_kernel[grid](
            a, aa, bb, d, 
            i, N, BLOCK_SIZE
        )