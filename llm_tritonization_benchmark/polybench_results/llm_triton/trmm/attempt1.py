import triton
import triton.language as tl
import torch

@triton.jit
def trmm_kernel(A_ptr, B_ptr, alpha, M: tl.constexpr, N: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    j_start = pid * BLOCK_N
    
    j_offsets = j_start + tl.arange(0, BLOCK_N)
    j_mask = j_offsets < N
    
    for i in range(M):
        # Load B[i, j_start:j_start+BLOCK_N]
        b_offsets = i * N + j_offsets
        b_vals = tl.load(B_ptr + b_offsets, mask=j_mask, other=0.0)
        
        # Accumulate over k from i+1 to M-1
        for k in range(i + 1, M):
            # Load A[k, i] (scalar)
            a_ki = tl.load(A_ptr + k * M + i)
            
            # Load B[k, j_start:j_start+BLOCK_N]
            b_k_offsets = k * N + j_offsets
            b_k_vals = tl.load(B_ptr + b_k_offsets, mask=j_mask, other=0.0)
            
            # Accumulate B[i, j] += A[k, i] * B[k, j]
            b_vals += a_ki * b_k_vals
        
        # Scale by alpha
        b_vals = alpha * b_vals
        
        # Store back B[i, j_start:j_start+BLOCK_N]
        tl.store(B_ptr + b_offsets, b_vals, mask=j_mask)

def trmm_triton(A, B, alpha, M, N):
    BLOCK_N = min(triton.next_power_of_2(N), 128)
    grid = (triton.cdiv(N, BLOCK_N),)
    
    trmm_kernel[grid](A, B, alpha, M, N, BLOCK_N)