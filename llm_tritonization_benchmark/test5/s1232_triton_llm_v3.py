import triton
import triton.language as tl
import torch

@triton.jit
def s1232_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID for j dimension
    j = tl.program_id(0)
    
    if j >= LEN_2D:
        return
    
    # Process elements where i >= j in blocks
    i_start = j
    i_offsets = tl.arange(0, BLOCK_SIZE)
    
    for i_block in range(i_start, LEN_2D, BLOCK_SIZE):
        i_indices = i_block + i_offsets
        mask = (i_indices < LEN_2D) & (i_indices >= j)
        
        if tl.sum(mask.to(tl.int32)) == 0:
            break
            
        # Calculate linear indices for 2D arrays
        linear_indices = i_indices * LEN_2D + j
        
        # Load bb and cc values
        bb_vals = tl.load(bb_ptr + linear_indices, mask=mask, other=0.0)
        cc_vals = tl.load(cc_ptr + linear_indices, mask=mask, other=0.0)
        
        # Compute result
        result = bb_vals + cc_vals
        
        # Store result
        tl.store(aa_ptr + linear_indices, result, mask=mask)

def s1232_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    
    BLOCK_SIZE = 128
    
    # Launch kernel with one program per j value
    grid = (LEN_2D,)
    
    s1232_kernel[grid](
        aa, bb, cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return aa