import triton
import triton.language as tl
import torch

@triton.jit
def syr2k_kernel(A, B, C, alpha, beta, M, N, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(0)
    
    if i >= N:
        return
    
    # First loop: multiply C[i][j] by beta for j <= i
    j_offsets = tl.arange(0, BLOCK_SIZE)
    for j_block_start in range(0, i + 1, BLOCK_SIZE):
        j_indices = j_block_start + j_offsets
        j_mask = (j_indices <= i) & (j_indices < N)
        
        c_ptr = C + i * N + j_indices
        c_vals = tl.load(c_ptr, mask=j_mask, other=0.0)
        c_vals = c_vals * beta
        tl.store(c_ptr, c_vals, mask=j_mask)
    
    # Second nested loop: accumulate into C[i][j]
    for k in range(M):
        j_offsets = tl.arange(0, BLOCK_SIZE)
        for j_block_start in range(0, i + 1, BLOCK_SIZE):
            j_indices = j_block_start + j_offsets
            j_mask = (j_indices <= i) & (j_indices < N)
            
            # Load A[j][k] and B[j][k]
            a_ptr = A + j_indices * M + k
            b_j_ptr = B + j_indices * M + k
            a_j_vals = tl.load(a_ptr, mask=j_mask, other=0.0)
            b_j_vals = tl.load(b_j_ptr, mask=j_mask, other=0.0)
            
            # Load B[i][k] and A[i][k]
            b_i_val = tl.load(B + i * M + k)
            a_i_val = tl.load(A + i * M + k)
            
            # Compute the update
            update = a_j_vals * alpha * b_i_val + b_j_vals * alpha * a_i_val
            
            # Load C[i][j], add update, and store back
            c_ptr = C + i * N + j_indices
            c_vals = tl.load(c_ptr, mask=j_mask, other=0.0)
            c_vals = c_vals + update
            tl.store(c_ptr, c_vals, mask=j_mask)

def syr2k_triton(A, B, C, alpha, beta, M, N):
    BLOCK_SIZE = 32
    grid = (N,)
    
    syr2k_kernel[grid](A, B, C, alpha, beta, M, N, BLOCK_SIZE)