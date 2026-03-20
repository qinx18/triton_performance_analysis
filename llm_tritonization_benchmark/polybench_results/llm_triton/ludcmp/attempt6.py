import triton
import triton.language as tl
import torch

@triton.jit
def ludcmp_decomposition_kernel(A_ptr, N: tl.constexpr):
    row = tl.program_id(0)
    
    # First loop: j < i (lower triangular part)
    for j in range(row):
        w = tl.load(A_ptr + row * N + j)
        for k in range(j):
            a_ik = tl.load(A_ptr + row * N + k)
            a_kj = tl.load(A_ptr + k * N + j)
            w = w - a_ik * a_kj
        a_jj = tl.load(A_ptr + j * N + j)
        result = w / a_jj
        tl.store(A_ptr + row * N + j, result)
    
    # Second loop: j >= i (upper triangular part)
    for j in range(row, N):
        w = tl.load(A_ptr + row * N + j)
        for k in range(row):
            a_ik = tl.load(A_ptr + row * N + k)
            a_kj = tl.load(A_ptr + k * N + j)
            w = w - a_ik * a_kj
        tl.store(A_ptr + row * N + j, w)

@triton.jit
def ludcmp_forward_kernel(A_ptr, b_ptr, y_ptr, N: tl.constexpr):
    row = tl.program_id(0)
    
    w = tl.load(b_ptr + row)
    for j in range(row):
        a_ij = tl.load(A_ptr + row * N + j)
        y_j = tl.load(y_ptr + j)
        w = w - a_ij * y_j
    tl.store(y_ptr + row, w)

@triton.jit
def ludcmp_backward_kernel(A_ptr, x_ptr, y_ptr, N: tl.constexpr):
    row = tl.program_id(0)
    i = N - 1 - row
    
    w = tl.load(y_ptr + i)
    for j in range(i + 1, N):
        a_ij = tl.load(A_ptr + i * N + j)
        x_j = tl.load(x_ptr + j)
        w = w - a_ij * x_j
    a_ii = tl.load(A_ptr + i * N + i)
    result = w / a_ii
    tl.store(x_ptr + i, result)

def ludcmp_triton(A, b, x, y, N):
    # LU decomposition phase
    grid = (N,)
    ludcmp_decomposition_kernel[grid](A, N)
    
    # Forward substitution phase
    ludcmp_forward_kernel[grid](A, b, y, N)
    
    # Backward substitution phase
    ludcmp_backward_kernel[grid](A, x, y, N)