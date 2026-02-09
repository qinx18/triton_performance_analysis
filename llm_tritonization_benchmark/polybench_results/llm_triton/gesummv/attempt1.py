import triton
import triton.language as tl
import torch

@triton.jit
def gesummv_kernel(A_ptr, B_ptr, tmp_ptr, x_ptr, y_ptr, alpha, beta, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    i_mask = i_offsets < N
    
    # Initialize tmp and y to 0
    tl.store(tmp_ptr + i_offsets, 0.0, mask=i_mask)
    tl.store(y_ptr + i_offsets, 0.0, mask=i_mask)
    
    # Inner loop over j
    for j in range(N):
        # Load x[j] (scalar for all threads)
        x_j = tl.load(x_ptr + j)
        
        # Load A[i][j] and B[i][j] for current rows
        A_indices = i_offsets * N + j
        B_indices = i_offsets * N + j
        
        A_vals = tl.load(A_ptr + A_indices, mask=i_mask, other=0.0)
        B_vals = tl.load(B_ptr + B_indices, mask=i_mask, other=0.0)
        
        # Load current tmp[i] and y[i]
        tmp_vals = tl.load(tmp_ptr + i_offsets, mask=i_mask, other=0.0)
        y_vals = tl.load(y_ptr + i_offsets, mask=i_mask, other=0.0)
        
        # Update tmp[i] and y[i]
        tmp_vals = A_vals * x_j + tmp_vals
        y_vals = B_vals * x_j + y_vals
        
        # Store updated values
        tl.store(tmp_ptr + i_offsets, tmp_vals, mask=i_mask)
        tl.store(y_ptr + i_offsets, y_vals, mask=i_mask)
    
    # Final computation: y[i] = alpha * tmp[i] + beta * y[i]
    tmp_vals = tl.load(tmp_ptr + i_offsets, mask=i_mask, other=0.0)
    y_vals = tl.load(y_ptr + i_offsets, mask=i_mask, other=0.0)
    y_final = alpha * tmp_vals + beta * y_vals
    tl.store(y_ptr + i_offsets, y_final, mask=i_mask)

def gesummv_triton(A, B, tmp, x, y, alpha, beta, N):
    BLOCK_SIZE = 64
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    gesummv_kernel[grid](
        A, B, tmp, x, y, 
        alpha, beta, N, 
        BLOCK_SIZE=BLOCK_SIZE
    )