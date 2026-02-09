import triton
import triton.language as tl
import torch

@triton.jit
def symm_kernel(A_ptr, B_ptr, C_ptr, alpha, beta, M, N, stride_A0, stride_A1, stride_B0, stride_B1, stride_C0, stride_C1, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Each program handles one (i, j) pair
    i = pid // N
    j = pid % N
    
    if i >= M or j >= N:
        return
    
    # Initialize temp2
    temp2 = 0.0
    
    # Load B[i][j] once since it's used multiple times
    B_ij = tl.load(B_ptr + i * stride_B0 + j * stride_B1)
    
    # Process k < i
    for k in range(i):
        # Load A[i][k]
        A_ik = tl.load(A_ptr + i * stride_A0 + k * stride_A1)
        
        # Load current C[k][j]
        C_kj = tl.load(C_ptr + k * stride_C0 + j * stride_C1)
        
        # Update C[k][j] += alpha * B[i][j] * A[i][k]
        C_kj_new = C_kj + alpha * B_ij * A_ik
        tl.store(C_ptr + k * stride_C0 + j * stride_C1, C_kj_new)
        
        # Load B[k][j] for temp2 accumulation  
        B_kj = tl.load(B_ptr + k * stride_B0 + j * stride_B1)
        
        # Accumulate temp2 += B[k][j] * A[i][k]
        temp2 += B_kj * A_ik
    
    # Load current C[i][j] after all updates to C[k][j] are done
    C_ij = tl.load(C_ptr + i * stride_C0 + j * stride_C1)
    
    # Load A[i][i]
    A_ii = tl.load(A_ptr + i * stride_A0 + i * stride_A1)
    
    # Update C[i][j] = beta * C[i][j] + alpha * B[i][j] * A[i][i] + alpha * temp2
    C_ij_new = beta * C_ij + alpha * B_ij * A_ii + alpha * temp2
    tl.store(C_ptr + i * stride_C0 + j * stride_C1, C_ij_new)

def symm_triton(A, B, C, alpha, beta, M, N):
    # Grid: one program per (i, j) pair
    grid = (M * N,)
    
    # Block size
    BLOCK_SIZE = 32
    
    # Make a copy of C to avoid race conditions
    C_work = C.clone()
    
    symm_kernel[grid](
        A, B, C_work,
        alpha, beta, M, N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C_work.stride(0), C_work.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Copy result back
    C.copy_(C_work)