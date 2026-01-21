import triton
import triton.language as tl
import torch

@triton.jit
def s256_kernel(a_ptr, aa_ptr, bb_ptr, d_ptr, i_val, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    j_offsets = offsets + 1
    mask = j_offsets < N
    
    prev_offsets = j_offsets - 1
    a_prev_vals = tl.load(a_ptr + prev_offsets, mask=mask)
    a_vals = 1.0 - a_prev_vals
    tl.store(a_ptr + j_offsets, a_vals, mask=mask)
    
    bb_vals = tl.load(bb_ptr + j_offsets * N + i_val, mask=mask)
    d_vals = tl.load(d_ptr + j_offsets, mask=mask)
    aa_vals = a_vals + bb_vals * d_vals
    tl.store(aa_ptr + j_offsets * N + i_val, aa_vals, mask=mask)

def s256_triton(a, aa, bb, d):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    for j in range(1, N):
        a[j] = 1.0 - a[j-1]
        
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        s256_kernel[grid](
            a, aa, bb, d, j, N, BLOCK_SIZE=BLOCK_SIZE
        )