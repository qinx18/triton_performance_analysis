import torch
import triton
import triton.language as tl

@triton.jit
def s114_kernel(aa_ptr, bb_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    
    for i in range(n):
        j_mask = j_idx < i
        j_mask = j_mask & (j_idx < n)
        
        if tl.sum(j_mask.to(tl.int32)) > 0:
            # Load aa[j, i] (transpose)
            aa_ji_offsets = j_idx * n + i
            aa_ji = tl.load(aa_ptr + aa_ji_offsets, mask=j_mask, other=0.0)
            
            # Load bb[i, j]
            bb_ij_offsets = i * n + j_idx
            bb_ij = tl.load(bb_ptr + bb_ij_offsets, mask=j_mask, other=0.0)
            
            # Compute aa[i, j] = aa[j, i] + bb[i, j]
            result = aa_ji + bb_ij
            
            # Store to aa[i, j]
            aa_ij_offsets = i * n + j_idx
            tl.store(aa_ptr + aa_ij_offsets, result, mask=j_mask)

def s114_triton(aa, bb):
    n = aa.shape[0]
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    s114_kernel[grid](aa, bb, n, BLOCK_SIZE=BLOCK_SIZE)