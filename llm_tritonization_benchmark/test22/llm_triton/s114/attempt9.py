import triton
import triton.language as tl
import torch

@triton.jit
def s114_kernel(aa_ptr, bb_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for i in range(N):
        j_start = pid * BLOCK_SIZE
        j_idx = j_start + j_offsets
        
        # Only process j values where j < i
        j_mask = (j_idx < i) & (j_idx >= 0) & (j_idx < N)
        
        # Skip if no valid j values in this block for this i
        if tl.sum(j_mask.to(tl.int32)) == 0:
            continue
            
        # aa[i][j] = aa[j][i] + bb[i][j]
        i_j_offsets = i * N + j_idx
        j_i_offsets = j_idx * N + i
        
        aa_j_i = tl.load(aa_ptr + j_i_offsets, mask=j_mask, other=0.0)
        bb_i_j = tl.load(bb_ptr + i_j_offsets, mask=j_mask, other=0.0)
        
        result = aa_j_i + bb_i_j
        
        tl.store(aa_ptr + i_j_offsets, result, mask=j_mask)

def s114_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s114_kernel[grid](aa, bb, N=N, BLOCK_SIZE=BLOCK_SIZE)