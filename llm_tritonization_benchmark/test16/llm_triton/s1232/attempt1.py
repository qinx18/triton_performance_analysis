import torch
import triton
import triton.language as tl

@triton.jit
def s1232_kernel(aa_ptr, bb_ptr, cc_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_base = pid * BLOCK_SIZE
    
    for j in range(N):
        i_idx = i_base + i_offsets
        
        # Only process elements where i >= j
        mask = (i_idx < N) & (i_idx >= j)
        
        # Calculate flat indices for [i, j] access
        flat_idx = i_idx * N + j
        
        bb_vals = tl.load(bb_ptr + flat_idx, mask=mask)
        cc_vals = tl.load(cc_ptr + flat_idx, mask=mask)
        
        result = bb_vals + cc_vals
        
        tl.store(aa_ptr + flat_idx, result, mask=mask)

def s1232_triton(aa, bb, cc):
    N = aa.shape[0]
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s1232_kernel[grid](aa, bb, cc, N, BLOCK_SIZE)