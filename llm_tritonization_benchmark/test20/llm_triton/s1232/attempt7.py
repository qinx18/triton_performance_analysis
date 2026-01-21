import torch
import triton
import triton.language as tl

@triton.jit
def s1232_kernel(aa_ptr, bb_ptr, cc_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_base = pid * BLOCK_SIZE
    i_offsets = tl.arange(0, BLOCK_SIZE)
    
    for j in range(N):
        i_indices = i_base + i_offsets
        
        # Only process elements where i >= j and i < N
        mask = (i_indices < N) & (i_indices >= j)
        
        # Calculate flat indices for 2D arrays
        flat_indices = i_indices * N + j
        
        # Load bb and cc values
        bb_vals = tl.load(bb_ptr + flat_indices, mask=mask, other=0.0)
        cc_vals = tl.load(cc_ptr + flat_indices, mask=mask, other=0.0)
        
        # Compute sum
        result = bb_vals + cc_vals
        
        # Store result
        tl.store(aa_ptr + flat_indices, result, mask=mask)

def s1232_triton(aa, bb, cc):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s1232_kernel[grid](
        aa, bb, cc, 
        N, BLOCK_SIZE
    )