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
        # Zero all elements in this block's range
        ptrs = aa_ptr + j_idx * N + i
        tl.store(ptrs, 0.0, mask=j_mask)
        
        # Set diagonal ONLY if i is in this block's j range
        diag_mask = j_mask & (j_idx == i)
        diag_ptrs = aa_ptr + j_idx * N + j_idx
        tl.store(diag_ptrs, 1.0, mask=diag_mask)

def s2102_triton(aa):
    N = aa.shape[0]
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s2102_kernel[grid](aa, N, BLOCK_SIZE)