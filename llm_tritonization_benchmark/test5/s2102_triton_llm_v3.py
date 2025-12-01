import torch
import triton
import triton.language as tl

@triton.jit
def s2102_kernel(aa_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID for the outer dimension (i)
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    # Define offsets once at the start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Zero out the entire column aa[*][i]
    for block_start in range(0, LEN_2D, BLOCK_SIZE):
        j_offsets = block_start + offsets
        mask = j_offsets < LEN_2D
        
        # Calculate pointer to aa[j][i] = aa_ptr[j * LEN_2D + i]
        ptr = aa_ptr + j_offsets * LEN_2D + i
        
        # Store zeros
        tl.store(ptr, 0.0, mask=mask)
    
    # Set diagonal element aa[i][i] = 1.0
    diagonal_ptr = aa_ptr + i * LEN_2D + i
    tl.store(diagonal_ptr, 1.0)

def s2102_triton(aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Launch kernel with one thread block per row
    grid = (LEN_2D,)
    
    s2102_kernel[grid](
        aa,
        LEN_2D,
        BLOCK_SIZE
    )
    
    return aa