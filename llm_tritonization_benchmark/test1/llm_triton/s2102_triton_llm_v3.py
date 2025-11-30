import triton
import triton.language as tl
import torch

@triton.jit
def s2102_kernel(aa_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID for the row
    i = tl.program_id(0)
    
    if i < LEN_2D:
        # Zero out the entire row
        for j_start in range(0, LEN_2D, BLOCK_SIZE):
            j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
            mask = j_offsets < LEN_2D
            
            # Calculate address for aa[j][i] (column-major indexing)
            addr = aa_ptr + j_offsets * LEN_2D + i
            
            # Store zeros
            tl.store(addr, 0.0, mask=mask)
        
        # Set diagonal element aa[i][i] = 1.0
        diag_addr = aa_ptr + i * LEN_2D + i
        tl.store(diag_addr, 1.0)

def s2102_triton(aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Launch kernel with one thread per row
    grid = (LEN_2D,)
    s2102_kernel[grid](aa, LEN_2D, BLOCK_SIZE)
    
    return aa