import triton
import triton.language as tl
import torch

@triton.jit
def s1232_kernel(aa_ptr, bb_ptr, cc_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_base = pid * BLOCK_SIZE
    i_idx = i_base + i_offsets
    
    for j in range(N):
        # Only process elements where i >= j and i < N
        valid_mask = (i_idx < N) & (i_idx >= j)
        
        # Calculate flat indices for 2D arrays
        flat_idx = j * N + i_idx
        
        # Load bb and cc values
        bb_vals = tl.load(bb_ptr + flat_idx, mask=valid_mask, other=0.0)
        cc_vals = tl.load(cc_ptr + flat_idx, mask=valid_mask, other=0.0)
        
        # Compute aa = bb + cc
        aa_vals = bb_vals + cc_vals
        
        # Store result
        tl.store(aa_ptr + flat_idx, aa_vals, mask=valid_mask)

def s1232_triton(aa, bb, cc):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s1232_kernel[grid](
        aa, bb, cc, N,
        BLOCK_SIZE=BLOCK_SIZE
    )