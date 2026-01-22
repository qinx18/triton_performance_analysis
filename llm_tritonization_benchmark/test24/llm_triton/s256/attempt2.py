import triton
import triton.language as tl
import torch

@triton.jit
def s256_kernel(aa_ptr, bb_ptr, d_ptr, a_val, N, BLOCK_SIZE: tl.constexpr):
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_mask = i_offsets < N
    
    # Load bb[j][i] values for all i
    bb_vals = tl.load(bb_ptr + i_offsets, mask=i_mask)
    
    # Load d[j]
    d_val = tl.load(d_ptr)
    
    # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
    aa_vals = a_val + bb_vals * d_val
    
    # Store aa[j][i] for all i
    tl.store(aa_ptr + i_offsets, aa_vals, mask=i_mask)

def s256_triton(a, aa, bb, d):
    N = aa.shape[0]  # LEN_2D
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    # Sequential loop over j from 1 to N-1
    for j in range(1, N):
        # Compute a[j] = 1.0 - a[j-1]
        a[j] = 1.0 - a[j-1]
        
        # Launch kernel for all i in parallel
        s256_kernel[grid](
            aa[j], bb[j], d + j, a[j].item(), N,
            BLOCK_SIZE=BLOCK_SIZE
        )