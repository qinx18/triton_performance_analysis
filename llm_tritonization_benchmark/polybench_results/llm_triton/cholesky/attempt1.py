import torch
import triton
import triton.language as tl

@triton.jit
def cholesky_kernel(A_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Get program IDs
    i = tl.program_id(0)
    
    if i >= N:
        return
    
    # Process j < i (lower triangular part)
    for j in range(i):
        # Compute A[i][j] -= sum(A[i][k] * A[j][k]) for k < j
        sum_val = 0.0
        for k in range(j):
            a_ik = tl.load(A_ptr + i * N + k)
            a_jk = tl.load(A_ptr + j * N + k)
            sum_val += a_ik * a_jk
        
        # Load current A[i][j], subtract sum, then divide by A[j][j]
        a_ij = tl.load(A_ptr + i * N + j)
        a_ij = a_ij - sum_val
        a_jj = tl.load(A_ptr + j * N + j)
        a_ij = a_ij / a_jj
        tl.store(A_ptr + i * N + j, a_ij)
    
    # Process diagonal case (i == j)
    # A[i][i] -= sum(A[i][k] * A[i][k]) for k < i
    sum_val = 0.0
    for k in range(i):
        a_ik = tl.load(A_ptr + i * N + k)
        sum_val += a_ik * a_ik
    
    # Load current A[i][i], subtract sum, then take square root
    a_ii = tl.load(A_ptr + i * N + i)
    a_ii = a_ii - sum_val
    a_ii = tl.sqrt(a_ii)
    tl.store(A_ptr + i * N + i, a_ii)

def cholesky_triton(A, N):
    # Launch kernel with one thread per row
    BLOCK_SIZE = 128
    grid = (N,)
    
    cholesky_kernel[grid](A, N, BLOCK_SIZE)