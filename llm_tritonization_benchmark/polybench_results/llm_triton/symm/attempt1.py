import triton
import triton.language as tl
import torch

@triton.jit
def symm_kernel(A_ptr, B_ptr, C_ptr, alpha, beta, M: tl.constexpr, N: tl.constexpr, BLOCK_N: tl.constexpr):
    j_block = tl.program_id(0)
    j_offsets = j_block * BLOCK_N + tl.arange(0, BLOCK_N)
    j_mask = j_offsets < N
    
    for i in range(M):
        # Load B[i][j] values for this i
        b_vals = tl.load(B_ptr + i * N + j_offsets, mask=j_mask, other=0.0)
        
        # Compute temp2 = sum(B[k][j] * A[i][k] for k in range(i))
        temp2_vals = tl.zeros([BLOCK_N], dtype=tl.float32)
        
        for k in range(i):
            # Load A[i][k] - this is scalar for each k
            a_ik = tl.load(A_ptr + i * M + k)
            
            # Load B[k][j] values
            b_k_vals = tl.load(B_ptr + k * N + j_offsets, mask=j_mask, other=0.0)
            
            # Update C[k][j] += alpha * B[i][j] * A[i][k]
            c_k_vals = tl.load(C_ptr + k * N + j_offsets, mask=j_mask, other=0.0)
            c_k_vals += alpha * b_vals * a_ik
            tl.store(C_ptr + k * N + j_offsets, c_k_vals, mask=j_mask)
            
            # Accumulate temp2 += B[k][j] * A[i][k]
            temp2_vals += b_k_vals * a_ik
        
        # Load A[i][i]
        a_ii = tl.load(A_ptr + i * M + i)
        
        # Load current C[i][j] values
        c_vals = tl.load(C_ptr + i * N + j_offsets, mask=j_mask, other=0.0)
        
        # Update C[i][j] = beta * C[i][j] + alpha * B[i][j] * A[i][i] + alpha * temp2
        c_vals = beta * c_vals + alpha * b_vals * a_ii + alpha * temp2_vals
        
        # Store updated C[i][j] values
        tl.store(C_ptr + i * N + j_offsets, c_vals, mask=j_mask)

def symm_triton(A, B, C, alpha, beta, M, N):
    BLOCK_N = min(triton.next_power_of_2(N), 128)
    grid = (triton.cdiv(N, BLOCK_N),)
    
    symm_kernel[grid](
        A, B, C, 
        alpha, beta,
        M, N, BLOCK_N
    )