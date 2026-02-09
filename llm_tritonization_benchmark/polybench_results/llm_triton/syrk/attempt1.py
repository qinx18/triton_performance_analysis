import triton
import triton.language as tl
import torch

@triton.jit
def syrk_kernel(A, C, alpha, beta, M, N, A_stride_0, A_stride_1, C_stride_0, C_stride_1, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i = pid
    
    if i >= N:
        return
    
    # First loop: C[i][j] *= beta for j <= i
    j_offsets = tl.arange(0, BLOCK_SIZE)
    for j_block in range(0, i + 1, BLOCK_SIZE):
        j_indices = j_block + j_offsets
        mask = (j_indices <= i) & (j_indices < N)
        
        c_ptrs = C + i * C_stride_0 + j_indices * C_stride_1
        c_vals = tl.load(c_ptrs, mask=mask)
        c_vals = c_vals * beta
        tl.store(c_ptrs, c_vals, mask=mask)
    
    # Second loop: C[i][j] += alpha * A[i][k] * A[j][k]
    for k in range(M):
        a_ik = tl.load(A + i * A_stride_0 + k * A_stride_1)
        
        for j_block in range(0, i + 1, BLOCK_SIZE):
            j_indices = j_block + j_offsets
            mask = (j_indices <= i) & (j_indices < N)
            
            a_jk_ptrs = A + j_indices * A_stride_0 + k * A_stride_1
            a_jk = tl.load(a_jk_ptrs, mask=mask)
            
            c_ptrs = C + i * C_stride_0 + j_indices * C_stride_1
            c_vals = tl.load(c_ptrs, mask=mask)
            c_vals = c_vals + alpha * a_ik * a_jk
            tl.store(c_ptrs, c_vals, mask=mask)

def syrk_triton(A, C, alpha, beta, M, N):
    BLOCK_SIZE = 64
    grid = (N,)
    
    syrk_kernel[grid](
        A, C, alpha, beta, M, N,
        A.stride(0), A.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE
    )