import triton
import triton.language as tl

@triton.jit
def ludcmp_lu_kernel(A, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    
    if row < N:
        # Process lower triangular part for this row
        for j in range(row):
            w = tl.load(A + row * N + j)
            for k in range(j):
                a_ik = tl.load(A + row * N + k)
                a_kj = tl.load(A + k * N + j)
                w -= a_ik * a_kj
            a_jj = tl.load(A + j * N + j)
            tl.store(A + row * N + j, w / a_jj)
        
        # Process upper triangular part for this row
        for j in range(row, N):
            w = tl.load(A + row * N + j)
            for k in range(row):
                a_ik = tl.load(A + row * N + k)
                a_kj = tl.load(A + k * N + j)
                w -= a_ik * a_kj
            tl.store(A + row * N + j, w)

@triton.jit
def ludcmp_forward_kernel(A, b, y, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    
    if row < N:
        w = tl.load(b + row)
        for j in range(row):
            a_ij = tl.load(A + row * N + j)
            y_j = tl.load(y + j)
            w -= a_ij * y_j
        tl.store(y + row, w)

@triton.jit
def ludcmp_backward_kernel(A, y, x, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    actual_row = N - 1 - row
    
    if actual_row >= 0:
        w = tl.load(y + actual_row)
        for j in range(actual_row + 1, N):
            a_ij = tl.load(A + actual_row * N + j)
            x_j = tl.load(x + j)
            w -= a_ij * x_j
        a_ii = tl.load(A + actual_row * N + actual_row)
        tl.store(x + actual_row, w / a_ii)

def ludcmp_triton(A, b, x, y, N):
    BLOCK_SIZE = 32
    
    # LU decomposition - each block handles one row
    grid = (N,)
    ludcmp_lu_kernel[grid](A, N, BLOCK_SIZE)
    
    # Forward substitution - each block handles one row
    grid = (N,)
    ludcmp_forward_kernel[grid](A, b, y, N, BLOCK_SIZE)
    
    # Backward substitution - each block handles one row (in reverse order)
    grid = (N,)
    ludcmp_backward_kernel[grid](A, y, x, N, BLOCK_SIZE)