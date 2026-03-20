import triton
import triton.language as tl

@triton.jit
def trmm_kernel(A_ptr, B_ptr, alpha, M: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(0)
    j = tl.program_id(1)
    
    if i < M and j < N:
        # Load initial B[i][j]
        b_val = tl.load(B_ptr + i * N + j)
        
        # Inner loop: for k in range(i+1, M)
        for k in range(i + 1, M):
            a_val = tl.load(A_ptr + k * M + i)
            b_k_j = tl.load(B_ptr + k * N + j)
            b_val += a_val * b_k_j
        
        # Apply alpha scaling
        b_val = alpha * b_val
        
        # Store result back to B[i][j]
        tl.store(B_ptr + i * N + j, b_val)

def trmm_triton(A, B, alpha, M, N):
    BLOCK_SIZE = 16
    
    grid = (M, N)
    
    trmm_kernel[grid](
        A, B, alpha,
        M, N, BLOCK_SIZE
    )