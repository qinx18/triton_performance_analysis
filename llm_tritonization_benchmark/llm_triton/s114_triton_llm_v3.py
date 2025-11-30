import triton
import triton.language as tl
import torch

@triton.jit
def s114_kernel(aa_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Sequential execution for both dimensions due to data dependencies
    for i in range(LEN_2D):
        for j in range(i):
            # Calculate memory addresses
            aa_ij_ptr = aa_ptr + i * LEN_2D + j
            aa_ji_ptr = aa_ptr + j * LEN_2D + i
            bb_ij_ptr = bb_ptr + i * LEN_2D + j
            
            # Load values
            aa_ji_val = tl.load(aa_ji_ptr)
            bb_ij_val = tl.load(bb_ij_ptr)
            
            # Compute and store
            result = aa_ji_val + bb_ij_val
            tl.store(aa_ij_ptr, result)

def s114_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 32
    
    # Launch kernel with single thread block since we need sequential execution
    grid = (1,)
    s114_kernel[grid](aa, bb, LEN_2D, BLOCK_SIZE)
    
    return aa