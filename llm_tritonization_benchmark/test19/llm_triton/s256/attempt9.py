import triton
import triton.language as tl
import torch

@triton.jit
def s256_kernel(aa_ptr, bb_ptr, d_ptr, a_val, i, N, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    mask = j_offsets < N
    
    aa_offsets = i * N + j_offsets
    bb_offsets = i * N + j_offsets
    d_offsets = j_offsets
    
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=mask)
    d_vals = tl.load(d_ptr + d_offsets, mask=mask)
    
    result = a_val + bb_vals * d_vals
    tl.store(aa_ptr + aa_offsets, result, mask=mask)

def s256_triton(a, aa, bb, d):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    for j in range(1, N):
        a[j] = 1.0 - a[j-1]
        
        for i in range(N):
            grid = (triton.cdiv(N, BLOCK_SIZE),)
            s256_kernel[grid](
                aa, 
                bb, 
                d,
                float(a[j]),
                i,
                N,
                BLOCK_SIZE=BLOCK_SIZE
            )