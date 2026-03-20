import triton
import triton.language as tl
import torch

@triton.jit
def syrk_kernel(A_ptr, C_ptr, alpha, beta, M: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Each program handles one row i
    i = tl.program_id(0)
    
    if i >= N:
        return
    
    # First loop: C[i][j] *= beta for j <= i
    j_offsets = tl.arange(0, BLOCK_SIZE)
    for j_start in range(0, i + 1, BLOCK_SIZE):
        current_j_offsets = j_start + j_offsets
        j_mask = (current_j_offsets < N) & (current_j_offsets <= i)
        
        c_idx = i * N + current_j_offsets
        c_vals = tl.load(C_ptr + c_idx, mask=j_mask)
        c_vals = c_vals * beta
        tl.store(C_ptr + c_idx, c_vals, mask=j_mask)
    
    # Second loop: accumulate alpha * A[i][k] * A[j][k]
    for k in range(M):
        a_ik = tl.load(A_ptr + i * M + k)
        
        for j_start in range(0, i + 1, BLOCK_SIZE):
            current_j_offsets = j_start + j_offsets
            j_mask = (current_j_offsets < N) & (current_j_offsets <= i)
            
            # Load A[j][k] values
            a_jk_idx = current_j_offsets * M + k
            a_jk_vals = tl.load(A_ptr + a_jk_idx, mask=j_mask)
            
            # Load current C[i][j] values
            c_idx = i * N + current_j_offsets
            c_vals = tl.load(C_ptr + c_idx, mask=j_mask)
            
            # Update C[i][j] += alpha * A[i][k] * A[j][k]
            c_vals = c_vals + alpha * a_ik * a_jk_vals
            tl.store(C_ptr + c_idx, c_vals, mask=j_mask)

def syrk_triton(A, C, alpha, beta, M, N):
    BLOCK_SIZE = 64
    
    grid = (N,)
    
    syrk_kernel[grid](
        A, C, alpha, beta, M, N, BLOCK_SIZE
    )