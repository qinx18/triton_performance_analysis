import torch
import triton
import triton.language as tl

@triton.jit
def s1232_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    
    for j in range(LEN_2D):
        i_idx = pid * BLOCK_SIZE + i_offsets
        
        # Mask for valid i indices: i >= j and i < LEN_2D
        mask = (i_idx >= j) & (i_idx < LEN_2D)
        
        # Calculate 2D indices: aa[i][j] -> i * LEN_2D + j
        indices_2d = i_idx * LEN_2D + j
        
        # Load values
        bb_vals = tl.load(bb_ptr + indices_2d, mask=mask, other=0.0)
        cc_vals = tl.load(cc_ptr + indices_2d, mask=mask, other=0.0)
        
        # Compute
        result = bb_vals + cc_vals
        
        # Store result
        tl.store(aa_ptr + indices_2d, result, mask=mask)

def s1232_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Flatten arrays for kernel access
    aa_flat = aa.flatten()
    bb_flat = bb.flatten()
    cc_flat = cc.flatten()
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s1232_kernel[grid](
        aa_flat, bb_flat, cc_flat,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )