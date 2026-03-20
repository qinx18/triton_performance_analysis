import triton
import triton.language as tl

@triton.jit
def ludcmp_lu_kernel(A, N: tl.constexpr, i: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid == 0:
        # Lower triangular part
        for j in range(i):
            w = tl.load(A + i * N + j)
            for k in range(j):
                a_ik = tl.load(A + i * N + k)
                a_kj = tl.load(A + k * N + j)
                w -= a_ik * a_kj
            a_jj = tl.load(A + j * N + j)
            tl.store(A + i * N + j, w / a_jj)
        
        # Upper triangular part
        for j in range(i, N):
            w = tl.load(A + i * N + j)
            for k in range(i):
                a_ik = tl.load(A + i * N + k)
                a_kj = tl.load(A + k * N + j)
                w -= a_ik * a_kj
            tl.store(A + i * N + j, w)

@triton.jit
def ludcmp_forward_kernel(A, b, y, N: tl.constexpr):
    row = tl.program_id(0)
    
    if row < N:
        w = tl.load(b + row)
        for j in range(row):
            a_ij = tl.load(A + row * N + j)
            y_j = tl.load(y + j)
            w -= a_ij * y_j
        tl.store(y + row, w)

@triton.jit
def ludcmp_backward_kernel(A, y, x, N: tl.constexpr, i: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid == 0:
        w = tl.load(y + i)
        for j in range(i + 1, N):
            a_ij = tl.load(A + i * N + j)
            x_j = tl.load(x + j)
            w -= a_ij * x_j
        a_ii = tl.load(A + i * N + i)
        tl.store(x + i, w / a_ii)

def ludcmp_triton(A, b, x, y, N):
    # LU decomposition
    for i in range(N):
        grid = (1,)
        ludcmp_lu_kernel[grid](A, N, i)
    
    # Forward substitution
    grid = (N,)
    ludcmp_forward_kernel[grid](A, b, y, N)
    
    # Backward substitution
    for i in range(N-1, -1, -1):
        grid = (1,)
        ludcmp_backward_kernel[grid](A, y, x, N, i)