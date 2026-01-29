import torch
import triton
import triton.language as tl

@triton.jit
def s114_kernel(aa_ptr, bb_ptr, n):
    BLOCK_SIZE: tl.constexpr = 256
    
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_base = pid * BLOCK_SIZE
    
    for i in range(n):
        j_idx = j_base + j_offsets
        
        # Only process j values where j < i
        j_mask = (j_idx < i) & (j_idx < n)
        
        # Load aa[j][i] for transpose access
        aa_ji_ptrs = aa_ptr + j_idx * n + i
        aa_ji_vals = tl.load(aa_ji_ptrs, mask=j_mask, other=0.0)
        
        # Load bb[i][j]
        bb_ij_ptrs = bb_ptr + i * n + j_idx
        bb_ij_vals = tl.load(bb_ij_ptrs, mask=j_mask, other=0.0)
        
        # Compute aa[i][j] = aa[j][i] + bb[i][j]
        result = aa_ji_vals + bb_ij_vals
        
        # Store to aa[i][j]
        aa_ij_ptrs = aa_ptr + i * n + j_idx
        tl.store(aa_ij_ptrs, result, mask=j_mask)

def s114_triton(aa, bb):
    n = aa.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s114_kernel[grid](aa, bb, n)