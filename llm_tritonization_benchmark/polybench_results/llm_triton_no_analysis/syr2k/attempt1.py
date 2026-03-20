import triton
import triton.language as tl
import torch

@triton.jit
def syr2k_kernel(A_ptr, B_ptr, C_ptr, alpha, beta, M: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID for the row
    i = tl.program_id(0)
    
    if i >= N:
        return
    
    # First, scale C[i][j] *= beta for j <= i
    j_offsets = tl.arange(0, BLOCK_SIZE)
    for j_start in range(0, i + 1, BLOCK_SIZE):
        current_j_offsets = j_start + j_offsets
        mask = (current_j_offsets < N) & (current_j_offsets <= i)
        
        c_indices = i * N + current_j_offsets
        c_vals = tl.load(C_ptr + c_indices, mask=mask)
        c_vals = c_vals * beta
        tl.store(C_ptr + c_indices, c_vals, mask=mask)
    
    # Accumulate C[i][j] += A[j][k]*alpha*B[i][k] + B[j][k]*alpha*A[i][k]
    for k in range(M):
        b_i_k = tl.load(B_ptr + i * M + k)  # B[i][k]
        a_i_k = tl.load(A_ptr + i * M + k)  # A[i][k]
        
        for j_start in range(0, i + 1, BLOCK_SIZE):
            current_j_offsets = j_start + j_offsets
            mask = (current_j_offsets < N) & (current_j_offsets <= i)
            
            # Load A[j][k] and B[j][k] for all j in current block
            a_indices = current_j_offsets * M + k
            b_indices = current_j_offsets * M + k
            
            a_j_k = tl.load(A_ptr + a_indices, mask=mask)  # A[j][k]
            b_j_k = tl.load(B_ptr + b_indices, mask=mask)  # B[j][k]
            
            # Compute the update
            update = a_j_k * alpha * b_i_k + b_j_k * alpha * a_i_k
            
            # Load current C[i][j] values and update
            c_indices = i * N + current_j_offsets
            c_vals = tl.load(C_ptr + c_indices, mask=mask)
            c_vals = c_vals + update
            tl.store(C_ptr + c_indices, c_vals, mask=mask)

def syr2k_triton(A, B, C, alpha, beta, M, N):
    BLOCK_SIZE = 32
    
    # Launch kernel with one thread block per row
    grid = (N,)
    
    syr2k_kernel[grid](
        A, B, C,
        alpha, beta,
        M, N, BLOCK_SIZE
    )
    
    return C