import torch
import triton
import triton.language as tl

@triton.jit
def s1232_kernel(aa_ptr, bb_ptr, cc_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < n
    
    for j in range(n):
        # Only process elements where i >= j
        valid_mask = i_mask & (i_idx >= j)
        
        if tl.sum(valid_mask.to(tl.int32)) > 0:
            # Calculate 2D indices for i,j access
            bb_indices = i_idx * n + j
            cc_indices = i_idx * n + j
            aa_indices = i_idx * n + j
            
            # Load bb[i][j] and cc[i][j] values
            bb_vals = tl.load(bb_ptr + bb_indices, mask=valid_mask, other=0.0)
            cc_vals = tl.load(cc_ptr + cc_indices, mask=valid_mask, other=0.0)
            
            # Compute aa[i][j] = bb[i][j] + cc[i][j]
            result = bb_vals + cc_vals
            
            # Store result back to aa[i][j]
            tl.store(aa_ptr + aa_indices, result, mask=valid_mask)

def s1232_triton(aa, bb, cc):
    n = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s1232_kernel[grid](
        aa, bb, cc,
        n,
        BLOCK_SIZE=BLOCK_SIZE
    )