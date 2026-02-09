import triton
import triton.language as tl
import torch

@triton.jit
def gemm_kernel(A, B, C, alpha, beta, NI, NJ, NK, BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    i_start = pid_i * BLOCK_I
    j_start = pid_j * BLOCK_J
    
    i_offsets = tl.arange(0, BLOCK_I)
    j_offsets = tl.arange(0, BLOCK_J)
    
    i_indices = i_start + i_offsets
    j_indices = j_start + j_offsets
    
    i_mask = i_indices < NI
    j_mask = j_indices < NJ
    
    # C[i][j] *= beta
    c_indices = i_indices[:, None] * NJ + j_indices[None, :]
    c_mask = i_mask[:, None] & j_mask[None, :]
    c_vals = tl.load(C + c_indices, mask=c_mask, other=0.0)
    c_vals = c_vals * beta
    
    # Accumulate A[i][k] * B[k][j] for all k
    for k in range(NK):
        a_indices = i_indices * NK + k
        a_vals = tl.load(A + a_indices, mask=i_mask, other=0.0)
        
        b_indices = k * NJ + j_indices
        b_vals = tl.load(B + b_indices, mask=j_mask, other=0.0)
        
        # Broadcast and multiply
        ab_product = alpha * a_vals[:, None] * b_vals[None, :]
        c_vals = c_vals + ab_product
    
    # Store result back to C
    tl.store(C + c_indices, c_vals, mask=c_mask)

def gemm_triton(A, B, C, alpha, beta, NI, NJ, NK):
    BLOCK_I = 16
    BLOCK_J = 16
    
    grid_i = triton.cdiv(NI, BLOCK_I)
    grid_j = triton.cdiv(NJ, BLOCK_J)
    
    gemm_kernel[(grid_i, grid_j)](
        A, B, C, alpha, beta, NI, NJ, NK, BLOCK_I, BLOCK_J
    )