import torch
import triton
import triton.language as tl

@triton.jit
def s1232_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_base = pid * BLOCK_SIZE
    
    for j in range(LEN_2D):
        i_idx = i_base + i_offsets
        
        # Only process elements where i >= j and i < LEN_2D
        mask = (i_idx >= j) & (i_idx < LEN_2D)
        
        # Calculate flat array offsets: aa[i][j] = flat[i * LEN_2D + j]
        offsets = i_idx * LEN_2D + j
        
        # Load data
        bb_vals = tl.load(bb_ptr + offsets, mask=mask, other=0.0)
        cc_vals = tl.load(cc_ptr + offsets, mask=mask, other=0.0)
        
        # Compute
        result = bb_vals + cc_vals
        
        # Store result
        tl.store(aa_ptr + offsets, result, mask=mask)

def s1232_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s1232_kernel[grid](
        aa, bb, cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )