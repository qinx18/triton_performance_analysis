import triton
import triton.language as tl
import torch

@triton.jit
def s114_kernel(aa_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get the current row index
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    # Process all j values for this i sequentially
    for j in range(i):
        # Calculate offsets
        aa_ij_offset = i * LEN_2D + j
        aa_ji_offset = j * LEN_2D + i
        bb_ij_offset = i * LEN_2D + j
        
        # Load values
        aa_ji_val = tl.load(aa_ptr + aa_ji_offset)
        bb_ij_val = tl.load(bb_ptr + bb_ij_offset)
        
        # Compute and store
        result = aa_ji_val + bb_ij_val
        tl.store(aa_ptr + aa_ij_offset, result)

def s114_triton(aa, bb):
    LEN_2D = aa.shape[0]
    
    # Launch kernel with one thread per row (i dimension)
    BLOCK_SIZE = 128
    grid = (LEN_2D,)
    
    s114_kernel[grid](
        aa, bb, 
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return aa