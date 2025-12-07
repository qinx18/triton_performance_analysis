import torch
import triton
import triton.language as tl

@triton.jit
def s1232_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    
    for j in range(LEN_2D):
        # Check bounds: i >= j and i < LEN_2D
        mask = (i_idx >= j) & (i_idx < LEN_2D)
        
        # Calculate linear indices for 2D arrays (row-major order)
        linear_idx = i_idx * LEN_2D + j
        
        # Load bb and cc values
        bb_vals = tl.load(bb_ptr + linear_idx, mask=mask)
        cc_vals = tl.load(cc_ptr + linear_idx, mask=mask)
        
        # Compute aa = bb + cc
        aa_vals = bb_vals + cc_vals
        
        # Store result
        tl.store(aa_ptr + linear_idx, aa_vals, mask=mask)

def s1232_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Flatten arrays for kernel access
    aa_flat = aa.view(-1)
    bb_flat = bb.view(-1)
    cc_flat = cc.view(-1)
    
    # Launch kernel
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    s1232_kernel[grid](
        aa_flat, bb_flat, cc_flat,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )