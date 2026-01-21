import triton
import triton.language as tl
import torch

@triton.jit
def s114_kernel(aa_ptr, bb_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    
    for i in range(N):
        j_mask = (j_idx < i) & (j_idx < N)
        
        if tl.sum(j_mask.to(tl.int32)) > 0:
            # Compute indices for aa[i][j]
            aa_write_idx = i * N + j_idx
            
            # Compute indices for aa[j][i]  
            aa_read_idx = j_idx * N + i
            
            # Compute indices for bb[i][j]
            bb_read_idx = i * N + j_idx
            
            # Load aa[j][i] and bb[i][j]
            aa_vals = tl.load(aa_ptr + aa_read_idx, mask=j_mask, other=0.0)
            bb_vals = tl.load(bb_ptr + bb_read_idx, mask=j_mask, other=0.0)
            
            # Compute result
            result = aa_vals + bb_vals
            
            # Store to aa[i][j]
            tl.store(aa_ptr + aa_write_idx, result, mask=j_mask)

def s114_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Launch kernel
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s114_kernel[grid](aa, bb, N, BLOCK_SIZE)