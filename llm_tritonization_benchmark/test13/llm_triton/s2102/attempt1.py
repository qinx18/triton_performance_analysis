import torch
import triton
import triton.language as tl

@triton.jit
def s2102_kernel(aa_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < LEN_2D
    
    for i in range(LEN_2D):
        # Zero out the entire column for this i
        aa_ptrs = aa_ptr + j_idx * LEN_2D + i
        zeros = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        tl.store(aa_ptrs, zeros, mask=j_mask)
        
        # Set diagonal element aa[i][i] = 1.0 (only one thread does this)
        if pid == 0:
            diagonal_ptr = aa_ptr + i * LEN_2D + i
            tl.store(diagonal_ptr, 1.0)

def s2102_triton(aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 128
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s2102_kernel[grid](
        aa,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )