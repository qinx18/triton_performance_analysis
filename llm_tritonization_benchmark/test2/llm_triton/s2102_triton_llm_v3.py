import torch
import triton
import triton.language as tl

@triton.jit
def s2102_kernel(aa_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID for the outer loop (i dimension)
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    # Zero out the entire column i
    for j_start in range(0, LEN_2D, BLOCK_SIZE):
        j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
        mask = j_offsets < LEN_2D
        
        # Calculate addresses for aa[j][i] = aa_ptr[j * LEN_2D + i]
        addrs = j_offsets * LEN_2D + i
        
        # Set all elements in this column to 0
        tl.store(aa_ptr + addrs, 0.0, mask=mask)
    
    # Set diagonal element aa[i][i] = 1.0
    diagonal_addr = i * LEN_2D + i
    tl.store(aa_ptr + diagonal_addr, 1.0)

def s2102_triton(aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Launch kernel with one thread block per row
    grid = (LEN_2D,)
    
    s2102_kernel[grid](
        aa,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return aa