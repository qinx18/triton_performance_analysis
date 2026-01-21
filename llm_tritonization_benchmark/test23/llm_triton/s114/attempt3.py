import torch
import triton
import triton.language as tl

@triton.jit
def s114_kernel(aa_ptr, bb_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_base = pid * BLOCK_SIZE
    
    for i in range(N):
        j_indices = j_base + j_offsets
        mask = j_indices < i
        
        if tl.sum(mask.to(tl.int32)) > 0:
            # Load aa[j, i] and bb[i, j]
            aa_ji_offsets = j_indices * N + i
            bb_ij_offsets = i * N + j_indices
            aa_ij_offsets = i * N + j_indices
            
            aa_ji = tl.load(aa_ptr + aa_ji_offsets, mask=mask, other=0.0)
            bb_ij = tl.load(bb_ptr + bb_ij_offsets, mask=mask, other=0.0)
            
            # Compute aa[i, j] = aa[j, i] + bb[i, j]
            result = aa_ji + bb_ij
            
            # Store to aa[i, j]
            tl.store(aa_ptr + aa_ij_offsets, result, mask=mask)

def s114_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate grid size for j dimension
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s114_kernel[grid](aa, bb, N, BLOCK_SIZE=BLOCK_SIZE)