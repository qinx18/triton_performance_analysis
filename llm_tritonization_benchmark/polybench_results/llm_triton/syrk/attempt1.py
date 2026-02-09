import triton
import triton.language as tl
import torch

@triton.jit
def syrk_kernel(A_ptr, C_ptr, alpha, beta, M, N, 
                A_stride0, A_stride1, C_stride0, C_stride1,
                BLOCK_SIZE: tl.constexpr):
    
    # Get program ID for the row
    i = tl.program_id(0)
    
    if i >= N:
        return
    
    # First loop: multiply C[i][j] by beta for j <= i
    j_offsets = tl.arange(0, BLOCK_SIZE)
    for j_start in range(0, i + 1, BLOCK_SIZE):
        current_j = j_start + j_offsets
        mask_j = (current_j < N) & (current_j <= i)
        
        c_ptrs = C_ptr + i * C_stride0 + current_j * C_stride1
        c_vals = tl.load(c_ptrs, mask=mask_j, other=0.0)
        c_vals = c_vals * beta
        tl.store(c_ptrs, c_vals, mask=mask_j)
    
    # Second loop: accumulate alpha * A[i][k] * A[j][k]
    for k in range(M):
        a_ik = tl.load(A_ptr + i * A_stride0 + k * A_stride1)
        
        for j_start in range(0, i + 1, BLOCK_SIZE):
            current_j = j_start + j_offsets
            mask_j = (current_j < N) & (current_j <= i)
            
            a_ptrs_j = A_ptr + current_j * A_stride0 + k * A_stride1
            a_jk = tl.load(a_ptrs_j, mask=mask_j, other=0.0)
            
            c_ptrs = C_ptr + i * C_stride0 + current_j * C_stride1
            c_vals = tl.load(c_ptrs, mask=mask_j, other=0.0)
            c_vals = c_vals + alpha * a_ik * a_jk
            tl.store(c_ptrs, c_vals, mask=mask_j)

def syrk_triton(A, C, alpha, beta, M, N):
    BLOCK_SIZE = 64
    grid = (triton.cdiv(N, 1),)
    
    syrk_kernel[grid](
        A, C, alpha, beta, M, N,
        A.stride(0), A.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )