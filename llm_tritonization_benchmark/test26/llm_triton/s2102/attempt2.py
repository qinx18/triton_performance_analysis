import torch
import triton
import triton.language as tl

@triton.jit
def s2102_kernel(aa_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < N

    for i in range(N):
        # Zero all elements in this block's j range for column i
        ptrs = aa_ptr + j_idx * N + i
        tl.store(ptrs, 0.0, mask=j_mask)
        
        # Set diagonal element if i is in this block's j range
        if i >= pid * BLOCK_SIZE and i < (pid + 1) * BLOCK_SIZE:
            diag_ptr = aa_ptr + i * N + i
            tl.store(diag_ptr, 1.0)

def s2102_triton(aa):
    N = aa.shape[0]
    BLOCK_SIZE = 128
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s2102_kernel[grid](aa, N, BLOCK_SIZE)