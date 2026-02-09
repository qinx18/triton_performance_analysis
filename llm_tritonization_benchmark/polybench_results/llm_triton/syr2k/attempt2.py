import triton
import triton.language as tl
import torch

@triton.jit
def syr2k_kernel(A, B, C, alpha, beta, M, N, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(0)
    
    if i >= N:
        return
    
    # First loop: C[i][j] *= beta for j <= i
    j_offsets = tl.arange(0, BLOCK_SIZE)
    num_j_blocks = (i + 1 + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    for j_block in range(num_j_blocks):
        j_indices = j_block * BLOCK_SIZE + j_offsets
        j_mask = (j_indices <= i) & (j_indices < N)
        
        c_ptrs = C + i * N + j_indices
        c_vals = tl.load(c_ptrs, mask=j_mask, other=0.0)
        c_vals = c_vals * beta
        tl.store(c_ptrs, c_vals, mask=j_mask)
    
    # Second nested loop: C[i][j] += A[j][k]*alpha*B[i][k] + B[j][k]*alpha*A[i][k]
    for k in range(M):
        b_i_k = tl.load(B + i * M + k)
        a_i_k = tl.load(A + i * M + k)
        
        for j_block in range(num_j_blocks):
            j_indices = j_block * BLOCK_SIZE + j_offsets
            j_mask = (j_indices <= i) & (j_indices < N)
            
            a_j_k_ptrs = A + j_indices * M + k
            b_j_k_ptrs = B + j_indices * M + k
            c_ptrs = C + i * N + j_indices
            
            a_j_k = tl.load(a_j_k_ptrs, mask=j_mask, other=0.0)
            b_j_k = tl.load(b_j_k_ptrs, mask=j_mask, other=0.0)
            c_vals = tl.load(c_ptrs, mask=j_mask, other=0.0)
            
            update = a_j_k * alpha * b_i_k + b_j_k * alpha * a_i_k
            c_vals = c_vals + update
            
            tl.store(c_ptrs, c_vals, mask=j_mask)

def syr2k_triton(A, B, C, alpha, beta, M, N):
    BLOCK_SIZE = 32
    grid = (triton.cdiv(N, 1),)
    
    syr2k_kernel[grid](A, B, C, alpha, beta, M, N, BLOCK_SIZE=BLOCK_SIZE)