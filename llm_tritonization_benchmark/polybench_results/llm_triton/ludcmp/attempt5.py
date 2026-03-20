import triton
import triton.language as tl

@triton.jit
def ludcmp_lu_kernel(A_ptr, row, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # LU decomposition for one row
    
    # Lower triangular part: j < row
    for j in range(row):
        w = tl.load(A_ptr + row * N + j)
        for k in range(j):
            w -= tl.load(A_ptr + row * N + k) * tl.load(A_ptr + k * N + j)
        tl.store(A_ptr + row * N + j, w / tl.load(A_ptr + j * N + j))
    
    # Upper triangular part: j >= row (vectorized)
    j_start = row
    j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
    
    for j_block in range(j_start, N, BLOCK_SIZE):
        current_j = j_block + tl.arange(0, BLOCK_SIZE)
        j_mask = current_j < N
        
        # Load A[row][j] for this block
        w_vec = tl.load(A_ptr + row * N + current_j, mask=j_mask)
        
        # Compute reduction: w -= A[row][k] * A[k][j] for k < row
        for k in range(row):
            a_row_k = tl.load(A_ptr + row * N + k)
            a_k_j = tl.load(A_ptr + k * N + current_j, mask=j_mask)
            w_vec -= a_row_k * a_k_j
        
        # Store result
        tl.store(A_ptr + row * N + current_j, w_vec, mask=j_mask)

@triton.jit 
def ludcmp_forward_kernel(A_ptr, b_ptr, y_ptr, row, N: tl.constexpr):
    # Forward substitution for one row
    w = tl.load(b_ptr + row)
    for j in range(row):
        w -= tl.load(A_ptr + row * N + j) * tl.load(y_ptr + j)
    tl.store(y_ptr + row, w)

@triton.jit
def ludcmp_backward_kernel(A_ptr, x_ptr, y_ptr, row, N: tl.constexpr):
    # Backward substitution for one row
    idx = N - 1 - row
    w = tl.load(y_ptr + idx)
    for j in range(idx + 1, N):
        w -= tl.load(A_ptr + idx * N + j) * tl.load(x_ptr + j)
    tl.store(x_ptr + idx, w / tl.load(A_ptr + idx * N + idx))

def ludcmp_triton(A, b, x, y, N):
    BLOCK_SIZE = 64
    
    # Phase 1: LU decomposition (sequential in rows)
    for i in range(N):
        ludcmp_lu_kernel[(1,)](A, i, N, BLOCK_SIZE)
    
    # Phase 2: Forward substitution (sequential in rows)  
    for i in range(N):
        ludcmp_forward_kernel[(1,)](A, b, y, i, N)
    
    # Phase 3: Backward substitution (sequential in rows)
    for i in range(N):
        ludcmp_backward_kernel[(1,)](A, x, y, i, N)