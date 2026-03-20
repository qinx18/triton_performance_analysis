import triton
import triton.language as tl
import torch

@triton.jit
def gesummv_kernel(A_ptr, B_ptr, tmp_ptr, x_ptr, y_ptr, alpha, beta, N: tl.constexpr):
    # Get program ID for the row
    row = tl.program_id(0)
    
    # Initialize accumulators
    tmp_val = 0.0
    y_val = 0.0
    
    # Inner loop over columns
    for j in range(N):
        # Load x[j]
        x_j = tl.load(x_ptr + j)
        
        # Load A[row, j] and B[row, j]
        a_val = tl.load(A_ptr + row * N + j)
        b_val = tl.load(B_ptr + row * N + j)
        
        # Accumulate
        tmp_val = a_val * x_j + tmp_val
        y_val = b_val * x_j + y_val
    
    # Store tmp[row]
    tl.store(tmp_ptr + row, tmp_val)
    
    # Compute final y[row]
    y_final = alpha * tmp_val + beta * y_val
    tl.store(y_ptr + row, y_final)

def gesummv_triton(A, B, tmp, x, y, alpha, beta, N):
    # Launch kernel with one thread per row
    grid = (N,)
    gesummv_kernel[grid](A, B, tmp, x, y, alpha, beta, N)