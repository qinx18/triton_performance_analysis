import triton
import triton.language as tl
import torch

@triton.jit
def syr2k_kernel(
    A_ptr, B_ptr, C_ptr,
    alpha, beta,
    M: tl.constexpr, N: tl.constexpr,
    BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr
):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    i_start = pid_i * BLOCK_I
    j_start = pid_j * BLOCK_J
    
    i_offsets = i_start + tl.arange(0, BLOCK_I)
    j_offsets = j_start + tl.arange(0, BLOCK_J)
    
    # Only process valid i indices
    i_mask = i_offsets < N
    
    for i_idx in range(BLOCK_I):
        i = i_start + i_idx
        if i >= N:
            break
            
        # Only process j <= i
        j_mask = (j_offsets <= i) & (j_offsets < N)
        
        if not tl.any(j_mask):
            continue
            
        # Phase 1: C[i][j] *= beta
        c_ptrs = C_ptr + i * N + j_offsets
        c_vals = tl.load(c_ptrs, mask=j_mask, other=0.0)
        c_vals = c_vals * beta
        tl.store(c_ptrs, c_vals, mask=j_mask)
        
        # Phase 2: accumulate over k
        for k in range(M):
            # Load A[j][k] and B[j][k] for valid j indices
            a_j_ptrs = A_ptr + j_offsets * M + k
            b_j_ptrs = B_ptr + j_offsets * M + k
            
            a_j_vals = tl.load(a_j_ptrs, mask=j_mask, other=0.0)
            b_j_vals = tl.load(b_j_ptrs, mask=j_mask, other=0.0)
            
            # Load A[i][k] and B[i][k] (scalars)
            a_i_val = tl.load(A_ptr + i * M + k)
            b_i_val = tl.load(B_ptr + i * M + k)
            
            # Compute update: A[j][k]*alpha*B[i][k] + B[j][k]*alpha*A[i][k]
            update = a_j_vals * alpha * b_i_val + b_j_vals * alpha * a_i_val
            
            # Load current C[i][j], add update, store back
            c_vals = tl.load(c_ptrs, mask=j_mask, other=0.0)
            c_vals = c_vals + update
            tl.store(c_ptrs, c_vals, mask=j_mask)

def syr2k_triton(A, B, C, alpha, beta, M, N):
    BLOCK_I = min(triton.next_power_of_2(N), 32)
    BLOCK_J = min(triton.next_power_of_2(N), 128)
    
    grid = (triton.cdiv(N, BLOCK_I), triton.cdiv(N, BLOCK_J))
    
    syr2k_kernel[grid](
        A, B, C,
        alpha, beta,
        M, N,
        BLOCK_I, BLOCK_J
    )