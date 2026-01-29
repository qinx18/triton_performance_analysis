import triton
import triton.language as tl
import torch

@triton.jit
def s1232_kernel(aa_ptr, bb_ptr, cc_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    
    for j in range(N):
        mask = (i_idx < N) & (i_idx >= j)
        
        bb_ptrs = bb_ptr + i_idx * N + j
        cc_ptrs = cc_ptr + i_idx * N + j
        aa_ptrs = aa_ptr + i_idx * N + j
        
        bb_vals = tl.load(bb_ptrs, mask=mask)
        cc_vals = tl.load(cc_ptrs, mask=mask)
        
        result = bb_vals + cc_vals
        
        tl.store(aa_ptrs, result, mask=mask)

def s1232_triton(aa, bb, cc):
    N = aa.shape[0]
    BLOCK_SIZE = 128
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s1232_kernel[grid](
        aa, bb, cc, 
        N, 
        BLOCK_SIZE
    )