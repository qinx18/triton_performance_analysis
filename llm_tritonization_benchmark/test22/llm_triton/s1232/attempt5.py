import triton
import triton.language as tl
import torch

@triton.jit
def s1232_kernel(aa_ptr, bb_ptr, cc_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_base = pid * BLOCK_SIZE
    
    for j in range(N):
        i_indices = i_base + i_offsets
        
        i_mask = (i_indices < N) & (i_indices >= j)
        
        flat_idx = i_indices * N + j
        
        bb_vals = tl.load(bb_ptr + flat_idx, mask=i_mask, other=0.0)
        cc_vals = tl.load(cc_ptr + flat_idx, mask=i_mask, other=0.0)
        
        result = bb_vals + cc_vals
        
        tl.store(aa_ptr + flat_idx, result, mask=i_mask)

def s1232_triton(aa, bb, cc):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Flatten the input tensors to 1D
    aa_flat = aa.view(-1)
    bb_flat = bb.view(-1)
    cc_flat = cc.view(-1)
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s1232_kernel[grid](
        aa_flat, bb_flat, cc_flat,
        N, BLOCK_SIZE
    )