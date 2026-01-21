import torch
import triton
import triton.language as tl

@triton.jit
def s114_kernel(aa, bb, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for i in range(N):
        j_start = pid * BLOCK_SIZE
        j_idx = j_start + j_offsets
        
        mask = (j_idx < i) & (j_idx < N)
        
        if tl.sum(mask.to(tl.int32)) > 0:
            aa_ji_vals = tl.load(aa + j_idx * N + i, mask=mask)
            bb_ij_vals = tl.load(bb + i * N + j_idx, mask=mask)
            result = aa_ji_vals + bb_ij_vals
            tl.store(aa + i * N + j_idx, result, mask=mask)

def s114_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 128
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s114_kernel[grid](aa, bb, N, BLOCK_SIZE=BLOCK_SIZE)