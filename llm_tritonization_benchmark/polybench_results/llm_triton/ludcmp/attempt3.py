import triton
import triton.language as tl

@triton.jit
def ludcmp_lu_kernel(A_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    
    if row >= N:
        return
        
    # Lower triangular part
    for j in range(row):
        w = tl.load(A_ptr + row * N + j)
        for k in range(j):
            w -= tl.load(A_ptr + row * N + k) * tl.load(A_ptr + k * N + j)
        tl.store(A_ptr + row * N + j, w / tl.load(A_ptr + j * N + j))
    
    # Upper triangular part - vectorized where possible
    j_offsets = tl.arange(0, BLOCK_SIZE)
    for j_start in range(row, N, BLOCK_SIZE):
        j_range = j_start + j_offsets
        j_mask = (j_range < N) & (j_range >= row)
        
        if tl.sum(j_mask.to(tl.int32)) == 0:
            continue
            
        # Load A[row, j_range]
        w_vec = tl.load(A_ptr + row * N + j_range, mask=j_mask, other=0.0)
        
        # Compute reduction for each j
        for k in range(row):
            a_ik = tl.load(A_ptr + row * N + k)
            a_kj_vec = tl.load(A_ptr + k * N + j_range, mask=j_mask, other=0.0)
            w_vec -= a_ik * a_kj_vec
        
        # Store result
        tl.store(A_ptr + row * N + j_range, w_vec, mask=j_mask)

@triton.jit
def ludcmp_forward_kernel(A_ptr, b_ptr, y_ptr, N: tl.constexpr):
    row = tl.program_id(0)
    
    if row >= N:
        return
        
    w = tl.load(b_ptr + row)
    for j in range(row):
        w -= tl.load(A_ptr + row * N + j) * tl.load(y_ptr + j)
    tl.store(y_ptr + row, w)

@triton.jit
def ludcmp_backward_kernel(A_ptr, x_ptr, y_ptr, N: tl.constexpr):
    row = tl.program_id(0)
    
    if row >= N:
        return
        
    i = N - 1 - row
    w = tl.load(y_ptr + i)
    for j in range(i + 1, N):
        w -= tl.load(A_ptr + i * N + j) * tl.load(x_ptr + j)
    tl.store(x_ptr + i, w / tl.load(A_ptr + i * N + i))

def ludcmp_triton(A, b, x, y, N):
    BLOCK_SIZE = min(triton.next_power_of_2(N), 128)
    
    # LU decomposition phase - parallelize across rows
    for i in range(N):
        ludcmp_lu_kernel[(N,)](A, N, BLOCK_SIZE)
    
    # Forward substitution - parallelize across rows
    ludcmp_forward_kernel[(N,)](A, b, y, N)
    
    # Backward substitution - parallelize across rows
    ludcmp_backward_kernel[(N,)](A, x, y, N)