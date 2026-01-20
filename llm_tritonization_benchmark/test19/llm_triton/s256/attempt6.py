import triton
import triton.language as tl
import torch

@triton.jit
def s256_kernel(a_ptr, aa_ptr, bb_ptr, d_ptr, N, BLOCK_SIZE: tl.constexpr):
    i_offsets = tl.arange(0, BLOCK_SIZE)
    mask = i_offsets < N
    
    bb_offsets = i_offsets * N
    aa_offsets = i_offsets * N
    
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=mask)
    d_val = tl.load(d_ptr)
    a_val = 1.0 - d_val
    
    result = a_val + bb_vals * d_val
    tl.store(aa_ptr + aa_offsets, result, mask=mask)

def s256_triton(a, aa, bb, d):
    N = aa.shape[1]
    BLOCK_SIZE = 256
    
    for j in range(1, N):
        a[j] = 1.0 - a[j-1]
        
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        s256_kernel[grid](
            a[j:j+1], 
            aa[j:j+1, :], 
            bb[j:j+1, :], 
            d[j:j+1],
            N,
            BLOCK_SIZE=BLOCK_SIZE
        )