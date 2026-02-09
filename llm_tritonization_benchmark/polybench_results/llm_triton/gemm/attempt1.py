import triton
import triton.language as tl
import torch

@triton.jit
def gemm_kernel(A, B, C, alpha, beta, NI, NJ, NK, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_J: tl.constexpr):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    i_offsets = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    j_offsets = pid_j * BLOCK_SIZE_J + tl.arange(0, BLOCK_SIZE_J)
    
    i_mask = i_offsets < NI
    j_mask = j_offsets < NJ
    
    # First multiply C[i][j] by beta
    c_ptrs = C + i_offsets[:, None] * NJ + j_offsets[None, :]
    mask_2d = i_mask[:, None] & j_mask[None, :]
    c_vals = tl.load(c_ptrs, mask=mask_2d, other=0.0)
    c_vals = c_vals * beta
    
    # Accumulate the matrix multiplication
    for k in range(NK):
        a_ptrs = A + i_offsets[:, None] * NK + k
        b_ptrs = B + k * NJ + j_offsets[None, :]
        
        a_vals = tl.load(a_ptrs, mask=i_mask[:, None], other=0.0)
        b_vals = tl.load(b_ptrs, mask=j_mask[None, :], other=0.0)
        
        c_vals += alpha * a_vals * b_vals
    
    tl.store(c_ptrs, c_vals, mask=mask_2d)

def gemm_triton(A, B, C, alpha, beta, NI, NJ, NK):
    BLOCK_SIZE_I = 16
    BLOCK_SIZE_J = 16
    
    grid = (triton.cdiv(NI, BLOCK_SIZE_I), triton.cdiv(NJ, BLOCK_SIZE_J))
    
    gemm_kernel[grid](
        A, B, C, alpha, beta, NI, NJ, NK,
        BLOCK_SIZE_I=BLOCK_SIZE_I,
        BLOCK_SIZE_J=BLOCK_SIZE_J
    )