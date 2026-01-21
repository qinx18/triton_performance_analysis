import triton
import triton.language as tl
import torch

@triton.jit
def s114_kernel(aa_ptr, bb_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_base = pid * BLOCK_SIZE
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_indices = j_base + j_offsets
    
    for i in range(N):
        # Only process j values where j < i
        j_mask = (j_indices < i) & (j_indices < N)
        
        if tl.sum(j_mask.to(tl.int32)) > 0:
            # Calculate addresses for aa[i][j], aa[j][i], bb[i][j]
            aa_ij_offsets = i * N + j_indices
            aa_ji_offsets = j_indices * N + i
            bb_ij_offsets = i * N + j_indices
            
            # Load values
            aa_ji_vals = tl.load(aa_ptr + aa_ji_offsets, mask=j_mask, other=0.0)
            bb_ij_vals = tl.load(bb_ptr + bb_ij_offsets, mask=j_mask, other=0.0)
            
            # Compute aa[i][j] = aa[j][i] + bb[i][j]
            result = aa_ji_vals + bb_ij_vals
            
            # Store result
            tl.store(aa_ptr + aa_ij_offsets, result, mask=j_mask)

def s114_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s114_kernel[grid](aa, bb, N, BLOCK_SIZE)