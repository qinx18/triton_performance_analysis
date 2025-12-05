import torch
import triton
import triton.language as tl

@triton.jit
def s114_kernel(aa_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Serial implementation due to data dependencies
    for i in range(LEN_2D):
        for j in range(i):
            # Calculate position in flattened 2D array
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
    BLOCK_SIZE = 64
    
    # Launch single block due to serial nature
    grid = (1,)
    
    s114_kernel[grid](
        aa, bb,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )