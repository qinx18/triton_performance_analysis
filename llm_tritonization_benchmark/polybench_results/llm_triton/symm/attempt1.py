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
    
    # Vectorized computation for k < i
    temp2 = 0.0
    k_offsets = tl.arange(0, BLOCK_SIZE)
    
    for k_start in range(0, i, BLOCK_SIZE):
        current_k = k_start + k_offsets
        k_mask = (current_k < i)
        
        # Load A[i, k] values
        A_ik_ptr = A_ptr + i * stride_A0 + current_k * stride_A1
        A_ik = tl.load(A_ik_ptr, mask=k_mask, other=0.0)
        
        # Load B[i, j] (scalar broadcast)
        B_ij_ptr = B_ptr + i * stride_B0 + j * stride_B1
        B_ij = tl.load(B_ij_ptr)
        
        # Load B[k, j] values
        B_kj_ptr = B_ptr + current_k * stride_B0 + j * stride_B1
        B_kj = tl.load(B_kj_ptr, mask=k_mask, other=0.0)
        
        # Load C[k, j] values
        C_kj_ptr = C_ptr + current_k * stride_C0 + j * stride_C1
        C_kj = tl.load(C_kj_ptr, mask=k_mask, other=0.0)
        
        # Update C[k, j] += alpha * B[i, j] * A[i, k]
        C_kj_new = C_kj + alpha * B_ij * A_ik
        tl.store(C_kj_ptr, C_kj_new, mask=k_mask)
        
        # Accumulate temp2 += B[k, j] * A[i, k]
        temp2_contrib = tl.sum(B_kj * A_ik, axis=0)
        temp2 += temp2_contrib
    
    # Load current C[i, j]
    C_ij_ptr = C_ptr + i * stride_C0 + j * stride_C1
    C_ij = tl.load(C_ij_ptr)
    
    # Load A[i, i]
    A_ii_ptr = A_ptr + i * stride_A0 + i * stride_A1
    A_ii = tl.load(A_ii_ptr)
    
    # Load B[i, j]
    B_ij_ptr = B_ptr + i * stride_B0 + j * stride_B1
    B_ij = tl.load(B_ij_ptr)
    
    # Update C[i, j]
    C_ij_new = beta * C_ij + alpha * B_ij * A_ii + alpha * temp2
    tl.store(C_ij_ptr, C_ij_new)

def symm_triton(A, B, C, alpha, beta, M, N):
    # Grid: one program per (i, j) pair
    grid = (M * N,)
    
    # Block size for vectorized k loop
    BLOCK_SIZE = 32
    
    symm_kernel[grid](
        A, B, C,
        alpha, beta, M, N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )