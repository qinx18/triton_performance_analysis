import triton
import triton.language as tl
import torch

@triton.jit
def s1232_kernel(aa_ptr, bb_ptr, cc_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_base = pid * BLOCK_SIZE
    
    for j in range(N):
        i_idx = i_base + i_offsets
        
        i_mask = (i_idx < N) & (i_idx >= j)
        
        idx = i_idx * N + j
        
        bb_vals = tl.load(bb_ptr + idx, mask=i_mask, other=0.0)
        cc_vals = tl.load(cc_ptr + idx, mask=i_mask, other=0.0)
        
        result = bb_vals + cc_vals
        
        tl.store(aa_ptr + idx, result, mask=i_mask)

def s1232_triton(aa, bb, cc):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s1232_kernel[grid](
        aa, bb, cc,
        N, BLOCK_SIZE
    )