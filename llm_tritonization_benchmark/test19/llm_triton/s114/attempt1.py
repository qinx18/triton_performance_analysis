import torch
import triton
import triton.language as tl

@triton.jit
def s114_kernel(aa_ptr, bb_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    
    for i in range(N):
        j_mask = (j_idx < i) & (j_idx < N)
        
        if tl.sum(j_mask) > 0:
            # Load aa[j, i] (transpose access)
            ji_offsets = j_idx * N + i
            aa_ji = tl.load(aa_ptr + ji_offsets, mask=j_mask)
            
            # Load bb[i, j]
            ij_offsets = i * N + j_idx
            bb_ij = tl.load(bb_ptr + ij_offsets, mask=j_mask)
            
            # Compute aa[i, j] = aa[j, i] + bb[i, j]
            result = aa_ji + bb_ij
            
            # Store to aa[i, j]
            tl.store(aa_ptr + ij_offsets, result, mask=j_mask)

def s114_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s114_kernel[grid](aa, bb, N=N, BLOCK_SIZE=BLOCK_SIZE)