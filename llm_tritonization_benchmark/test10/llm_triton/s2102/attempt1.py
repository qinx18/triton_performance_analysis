import triton
import triton.language as tl
import torch

@triton.jit
def s2102_kernel(aa_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < LEN_2D
    
    for i in range(LEN_2D):
        # Zero out column i for all j values in this block
        col_ptrs = aa_ptr + j_idx * LEN_2D + i
        tl.store(col_ptrs, 0.0, mask=j_mask)
        
        # Set diagonal element aa[i][i] = 1.0 if it's in our block
        if (i >= pid * BLOCK_SIZE) & (i < (pid + 1) * BLOCK_SIZE):
            diag_ptr = aa_ptr + i * LEN_2D + i
            tl.store(diag_ptr, 1.0)

def s2102_triton(aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    s2102_kernel[grid](aa, LEN_2D, BLOCK_SIZE)