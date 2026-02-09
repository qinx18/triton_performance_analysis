import torch
import triton
import triton.language as tl

@triton.jit
def atax_kernel(A_ptr, tmp_ptr, x_ptr, y_ptr, M, N):
    pid = tl.program_id(0)
    
    BLOCK_SIZE = 64
    
    if pid < M:
        # Each program handles one row i
        i = pid
        
        # Initialize tmp[i] = 0.0
        tmp_val = 0.0
        
        # First loop: tmp[i] += A[i][j] * x[j] for all j
        x_offsets = tl.arange(0, BLOCK_SIZE)
        j_start = 0
        while j_start < N:
            current_x_offsets = j_start + x_offsets
            x_mask = current_x_offsets < N
            
            # Load x[j]
            x_vals = tl.load(x_ptr + current_x_offsets, mask=x_mask, other=0.0)
            
            # Load A[i][j]
            a_offsets = i * N + current_x_offsets
            a_vals = tl.load(A_ptr + a_offsets, mask=x_mask, other=0.0)
            
            # Accumulate tmp[i]
            tmp_val += tl.sum(a_vals * x_vals)
            j_start += BLOCK_SIZE
        
        # Store tmp[i]
        tl.store(tmp_ptr + i, tmp_val)
        
        # Second loop: y[j] += A[i][j] * tmp[i] for all j
        y_offsets = tl.arange(0, BLOCK_SIZE)
        j_start = 0
        while j_start < N:
            current_y_offsets = j_start + y_offsets
            y_mask = current_y_offsets < N
            
            # Load current y[j]
            y_vals = tl.load(y_ptr + current_y_offsets, mask=y_mask, other=0.0)
            
            # Load A[i][j]
            a_offsets = i * N + current_y_offsets
            a_vals = tl.load(A_ptr + a_offsets, mask=y_mask, other=0.0)
            
            # Update y[j]
            y_vals += a_vals * tmp_val
            tl.store(y_ptr + current_y_offsets, y_vals, mask=y_mask)
            j_start += BLOCK_SIZE

@triton.jit
def init_y_kernel(y_ptr, N):
    pid = tl.program_id(0)
    BLOCK_SIZE = 64
    
    y_start = pid * BLOCK_SIZE
    y_offsets = tl.arange(0, BLOCK_SIZE)
    current_y_offsets = y_start + y_offsets
    y_mask = current_y_offsets < N
    
    tl.store(y_ptr + current_y_offsets, 0.0, mask=y_mask)

def atax_triton(A, tmp, x, y, M, N):
    # Initialize y to zero
    grid_init = (triton.cdiv(N, 64),)
    init_y_kernel[grid_init](y, N)
    
    # Main computation
    grid = (M,)
    atax_kernel[grid](A, tmp, x, y, M, N)