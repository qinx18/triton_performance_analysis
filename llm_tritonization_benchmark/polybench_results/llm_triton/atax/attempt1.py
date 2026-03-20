import triton
import triton.language as tl
import torch

@triton.jit
def atax_kernel(A_ptr, x_ptr, y_ptr, tmp_ptr, M: tl.constexpr, N: tl.constexpr, BLOCK_N: tl.constexpr):
    # This is a fused kernel that handles both reductions in a single pass
    # Iterate over rows (M dimension) and vectorize columns (N dimension)
    
    offsets_n = tl.arange(0, BLOCK_N)
    mask_n = offsets_n < N
    
    # Initialize y to zero
    y_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    
    # Iterate over all rows
    for i in range(M):
        # Load row i of matrix A
        a_row_ptr = A_ptr + i * N + offsets_n
        a_row = tl.load(a_row_ptr, mask=mask_n, other=0.0)
        
        # Load x vector
        x_vec = tl.load(x_ptr + offsets_n, mask=mask_n, other=0.0)
        
        # Compute tmp[i] = sum(A[i][j] * x[j]) - row reduction
        tmp_val = tl.sum(a_row * x_vec)
        tl.store(tmp_ptr + i, tmp_val)
        
        # Accumulate y[j] += A[i][j] * tmp[i] - column accumulation
        y_acc += a_row * tmp_val
    
    # Store final y values
    tl.store(y_ptr + offsets_n, y_acc, mask=mask_n)

def atax_triton(A, tmp, x, y, M, N):
    # Initialize y to zero
    y.zero_()
    
    # Use single block to handle all N elements
    BLOCK_N = triton.next_power_of_2(N)
    
    # Single kernel launch with grid=(1,) to fuse both reductions
    grid = (1,)
    
    atax_kernel[grid](
        A, x, y, tmp,
        M=M, N=N, BLOCK_N=BLOCK_N
    )