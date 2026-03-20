import triton
import triton.language as tl
import torch

@triton.jit
def lu_kernel(A_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    
    if row >= N:
        return
    
    # First loop: for j = 0 to row-1
    for j in range(row):
        # Inner reduction: A[row][j] -= sum(A[row][k] * A[k][j] for k in 0..j-1)
        for k in range(j):
            row_k_offset = row * N + k
            k_j_offset = k * N + j
            row_j_offset = row * N + j
            
            a_row_k = tl.load(A_ptr + row_k_offset)
            a_k_j = tl.load(A_ptr + k_j_offset)
            a_row_j = tl.load(A_ptr + row_j_offset)
            
            a_row_j = a_row_j - a_row_k * a_k_j
            tl.store(A_ptr + row_j_offset, a_row_j)
        
        # A[row][j] /= A[j][j]
        row_j_offset = row * N + j
        j_j_offset = j * N + j
        
        a_row_j = tl.load(A_ptr + row_j_offset)
        a_j_j = tl.load(A_ptr + j_j_offset)
        
        a_row_j = a_row_j / a_j_j
        tl.store(A_ptr + row_j_offset, a_row_j)
    
    # Second loop: for j = row to N-1
    for j in range(row, N):
        # Inner reduction: A[row][j] -= sum(A[row][k] * A[k][j] for k in 0..row-1)
        for k in range(row):
            row_k_offset = row * N + k
            k_j_offset = k * N + j
            row_j_offset = row * N + j
            
            a_row_k = tl.load(A_ptr + row_k_offset)
            a_k_j = tl.load(A_ptr + k_j_offset)
            a_row_j = tl.load(A_ptr + row_j_offset)
            
            a_row_j = a_row_j - a_row_k * a_k_j
            tl.store(A_ptr + row_j_offset, a_row_j)

def lu_triton(A, N):
    BLOCK_SIZE = 32
    grid = (N,)
    lu_kernel[grid](A, N, BLOCK_SIZE)