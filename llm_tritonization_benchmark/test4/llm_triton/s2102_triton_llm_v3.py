import torch
import triton
import triton.language as tl

@triton.jit
def s2102_kernel(aa_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID for the outer loop (i dimension)
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    # Define offsets once at kernel start
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Zero out the entire column aa[:][i] in blocks
    for j_block_start in range(0, LEN_2D, BLOCK_SIZE):
        j_indices = j_block_start + j_offsets
        j_mask = j_indices < LEN_2D
        
        # Calculate memory addresses: aa[j][i] = aa_ptr[j * LEN_2D + i]
        addr = j_indices * LEN_2D + i
        tl.store(aa_ptr + addr, 0.0, mask=j_mask)
    
    # Set diagonal element aa[i][i] = 1.0
    diagonal_addr = i * LEN_2D + i
    tl.store(aa_ptr + diagonal_addr, 1.0)

def s2102_triton(aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Launch kernel with one thread block per row (i dimension)
    grid = (LEN_2D,)
    
    s2102_kernel[grid](
        aa,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )