import triton
import triton.language as tl
import torch

@triton.jit
def gemm_kernel(A, B, C, alpha, beta, NI: tl.constexpr, NJ: tl.constexpr, NK: tl.constexpr, 
                BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    block_start_i = pid_i * BLOCK_I
    block_start_j = pid_j * BLOCK_J
    
    offs_i = block_start_i + tl.arange(0, BLOCK_I)
    offs_j = block_start_j + tl.arange(0, BLOCK_J)
    
    mask_i = offs_i < NI
    mask_j = offs_j < NJ
    
    # Load and scale C[i, j] by beta
    c_offs = offs_i[:, None] * NJ + offs_j[None, :]
    mask_c = mask_i[:, None] & mask_j[None, :]
    c_vals = tl.load(C + c_offs, mask=mask_c, other=0.0)
    c_vals = c_vals * beta
    
    # Compute matrix multiplication: C += alpha * A * B
    for k in range(NK):
        a_offs = offs_i * NK + k
        b_offs = k * NJ + offs_j
        
        a_vals = tl.load(A + a_offs, mask=mask_i, other=0.0)
        b_vals = tl.load(B + b_offs, mask=mask_j, other=0.0)
        
        c_vals += alpha * a_vals[:, None] * b_vals[None, :]
    
    # Store result back to C
    tl.store(C + c_offs, c_vals, mask=mask_c)

def gemm_triton(A, B, C, alpha, beta, NI, NJ, NK):
    BLOCK_I = 32
    BLOCK_J = 32
    
    grid = (triton.cdiv(NI, BLOCK_I), triton.cdiv(NJ, BLOCK_J))
    
    gemm_kernel[grid](
        A, B, C, alpha, beta, NI, NJ, NK, BLOCK_I, BLOCK_J
    )