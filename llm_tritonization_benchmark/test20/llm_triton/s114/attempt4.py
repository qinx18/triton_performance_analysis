import torch
import triton
import triton.language as tl

@triton.jit
def s114_kernel(aa_ptr, bb_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for i in range(N):
        j_base = pid * BLOCK_SIZE
        j_indices = j_base + j_offsets
        
        # Only process j < i
        mask = (j_indices < i) & (j_indices < N)
        
        if tl.sum(mask.to(tl.int32)) > 0:
            # Load aa[j][i] (transpose read)
            transpose_indices = j_indices * N + i
            aa_ji = tl.load(aa_ptr + transpose_indices, mask=mask, other=0.0)
            
            # Load bb[i][j]
            bb_indices = i * N + j_indices
            bb_ij = tl.load(bb_ptr + bb_indices, mask=mask, other=0.0)
            
            # Compute aa[i][j] = aa[j][i] + bb[i][j]
            result = aa_ji + bb_ij
            
            # Store to aa[i][j]
            store_indices = i * N + j_indices
            tl.store(aa_ptr + store_indices, result, mask=mask)

def s114_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s114_kernel[grid](aa, bb, N=N, BLOCK_SIZE=BLOCK_SIZE)