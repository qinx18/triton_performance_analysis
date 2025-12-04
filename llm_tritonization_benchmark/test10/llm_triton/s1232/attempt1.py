import torch
import triton
import triton.language as tl

@triton.jit
def s1232_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    
    for j in range(LEN_2D):
        # Current indices for this j iteration
        current_i_idx = i_idx
        
        # Mask for valid indices (i >= j and i < LEN_2D)
        mask = (current_i_idx >= j) & (current_i_idx < LEN_2D)
        
        # Calculate linear indices for 2D arrays
        linear_idx = current_i_idx * LEN_2D + j
        
        # Load values
        bb_vals = tl.load(bb_ptr + linear_idx, mask=mask)
        cc_vals = tl.load(cc_ptr + linear_idx, mask=mask)
        
        # Compute
        result = bb_vals + cc_vals
        
        # Store result
        tl.store(aa_ptr + linear_idx, result, mask=mask)

def s1232_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 128
    
    # Launch kernel with single grid dimension for i
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s1232_kernel[grid](
        aa, bb, cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )