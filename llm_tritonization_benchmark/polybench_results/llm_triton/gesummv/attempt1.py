import triton
import triton.language as tl
import torch

@triton.jit
def gesummv_kernel(
    A_ptr, B_ptr, tmp_ptr, x_ptr, y_ptr, 
    alpha, beta,
    N: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    i = tl.program_id(0)
    
    if i >= N:
        return
    
    # Initialize accumulation values
    tmp_acc = 0.0
    y_acc = 0.0
    
    # Get j offsets once
    j_offsets = tl.arange(0, BLOCK_J)
    
    # Reduction over j dimension
    for j_start in range(0, N, BLOCK_J):
        j_idx = j_start + j_offsets
        j_mask = j_idx < N
        
        # Load A[i, j] and B[i, j]
        A_offsets = i * N + j_idx
        B_offsets = i * N + j_idx
        
        A_vals = tl.load(A_ptr + A_offsets, mask=j_mask, other=0.0)
        B_vals = tl.load(B_ptr + B_offsets, mask=j_mask, other=0.0)
        x_vals = tl.load(x_ptr + j_idx, mask=j_mask, other=0.0)
        
        # Accumulate tmp[i] and y[i]
        tmp_acc += tl.sum(A_vals * x_vals)
        y_acc += tl.sum(B_vals * x_vals)
    
    # Final computation: y[i] = alpha * tmp[i] + beta * y[i]
    final_y = alpha * tmp_acc + beta * y_acc
    
    # Store results
    tl.store(tmp_ptr + i, tmp_acc)
    tl.store(y_ptr + i, final_y)

def gesummv_triton(A, B, tmp, x, y, alpha, beta, N):
    BLOCK_J = 64
    
    grid = (N,)
    
    gesummv_kernel[grid](
        A, B, tmp, x, y,
        alpha, beta,
        N=N,
        BLOCK_J=BLOCK_J,
    )