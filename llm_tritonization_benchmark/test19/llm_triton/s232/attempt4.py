import triton
import triton.language as tl
import torch

@triton.jit
def s232_kernel(aa_ptr, bb_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < N
    
    # Start from j=1 (skip j=0)
    valid_j_mask = j_mask & (j_idx >= 1)
    
    for i in range(1, N):
        # Only process j values where j >= i (triangular condition)
        triangular_mask = valid_j_mask & (j_idx >= i)
        
        # Load aa[j][i-1] 
        aa_prev_ptrs = aa_ptr + j_idx * N + (i - 1)
        aa_prev = tl.load(aa_prev_ptrs, mask=triangular_mask, other=0.0)
        
        # Load bb[j][i]
        bb_ptrs = bb_ptr + j_idx * N + i
        bb_vals = tl.load(bb_ptrs, mask=triangular_mask, other=0.0)
        
        # Compute aa[j][i] = aa[j][i-1] * aa[j][i-1] + bb[j][i]
        result = aa_prev * aa_prev + bb_vals
        
        # Store aa[j][i]
        aa_ptrs = aa_ptr + j_idx * N + i
        tl.store(aa_ptrs, result, mask=triangular_mask)

def s232_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s232_kernel[grid](aa, bb, N, BLOCK_SIZE=BLOCK_SIZE)