import triton
import triton.language as tl
import torch

@triton.jit
def cholesky_kernel(A_ptr, N: tl.constexpr, i: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j = tl.program_id(0)
    
    if j >= i:
        return
    
    # Compute A[i][j] -= sum(A[i][k] * A[j][k] for k < j)
    sum_val = 0.0
    for k in range(j):
        a_ik = tl.load(A_ptr + i * N + k)
        a_jk = tl.load(A_ptr + j * N + k)
        sum_val += a_ik * a_jk
    
    # Update A[i][j]
    a_ij_offset = i * N + j
    a_ij = tl.load(A_ptr + a_ij_offset)
    a_ij -= sum_val
    
    # Divide by A[j][j]
    a_jj = tl.load(A_ptr + j * N + j)
    a_ij /= a_jj
    
    # Store updated A[i][j]
    tl.store(A_ptr + a_ij_offset, a_ij)

@triton.jit
def cholesky_diagonal_kernel(A_ptr, N: tl.constexpr, i: tl.constexpr):
    # Process diagonal element A[i][i]
    sum_val = 0.0
    for k in range(i):
        a_ik = tl.load(A_ptr + i * N + k)
        sum_val += a_ik * a_ik
    
    # Update A[i][i]
    a_ii_offset = i * N + i
    a_ii = tl.load(A_ptr + a_ii_offset)
    a_ii -= sum_val
    a_ii = tl.sqrt(a_ii)
    
    # Store updated A[i][i]
    tl.store(A_ptr + a_ii_offset, a_ii)

def cholesky_triton(A, N):
    BLOCK_SIZE = 128
    
    for i in range(N):
        if i > 0:
            # Process off-diagonal elements in parallel
            grid = (triton.cdiv(i, BLOCK_SIZE),)
            cholesky_kernel[grid](
                A, N, i, BLOCK_SIZE
            )
        
        # Process diagonal element
        grid = (1,)
        cholesky_diagonal_kernel[grid](
            A, N, i
        )