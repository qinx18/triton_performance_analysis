import triton
import triton.language as tl
import torch

@triton.jit
def s2102_kernel(aa_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < N

    for i in range(N):
        # Zero all elements aa[j][i] for this block's j range
        ptrs = aa_ptr + j_idx * N + i
        tl.store(ptrs, 0.0, mask=j_mask)

    # After all columns are zeroed, set diagonal elements
    for i in range(N):
        diag_mask = j_mask & (j_idx == i)
        if tl.any(diag_mask):
            diag_ptrs = aa_ptr + i * N + i
            tl.store(diag_ptrs, 1.0)

def s2102_triton(aa):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s2102_kernel[grid](aa, N, BLOCK_SIZE)