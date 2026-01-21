import torch
import triton
import triton.language as tl

@triton.jit
def s114_kernel(aa_ptr, bb_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_start = pid * BLOCK_SIZE
    
    for i in range(1, N):
        j_indices = j_start + j_offsets
        
        # Mask for valid j values: j < i
        j_mask = (j_indices < i) & (j_indices >= 0)
        
        # Load aa[j][i] (transpose indices)
        ji_indices = j_indices * N + i
        aa_ji_vals = tl.load(aa_ptr + ji_indices, mask=j_mask, other=0.0)
        
        # Load bb[i][j]
        ij_indices = i * N + j_indices
        bb_ij_vals = tl.load(bb_ptr + ij_indices, mask=j_mask, other=0.0)
        
        # Compute aa[i][j] = aa[j][i] + bb[i][j]
        result_vals = aa_ji_vals + bb_ij_vals
        
        # Store to aa[i][j]
        tl.store(aa_ptr + ij_indices, result_vals, mask=j_mask)

def s114_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate grid size based on maximum j values needed
    max_j = N - 1  # j ranges from 0 to N-2 (since j < i and i goes up to N-1)
    grid = (triton.cdiv(max_j, BLOCK_SIZE),)
    
    s114_kernel[grid](aa, bb, N=N, BLOCK_SIZE=BLOCK_SIZE)