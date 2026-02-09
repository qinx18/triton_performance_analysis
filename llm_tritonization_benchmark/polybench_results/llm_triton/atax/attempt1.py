import torch
import triton
import triton.language as tl

@triton.jit
def atax_kernel(A_ptr, tmp_ptr, x_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Initialize y to zero
    if pid == 0:
        y_offsets = tl.arange(0, BLOCK_SIZE)
        for y_start in range(0, N, BLOCK_SIZE):
            y_current_offsets = y_start + y_offsets
            y_mask = y_current_offsets < N
            tl.store(y_ptr + y_current_offsets, 0.0, mask=y_mask)
    
    # Synchronize to ensure y is initialized
    tl.debug_barrier()
    
    # Each program handles one row of A
    i = pid
    if i >= M:
        return
    
    # Initialize tmp[i] = 0.0
    tl.store(tmp_ptr + i, 0.0)
    
    # Compute tmp[i] = sum(A[i][j] * x[j]) for j in [0, N)
    tmp_val = 0.0
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for j_start in range(0, N, BLOCK_SIZE):
        j_offsets = j_start + offsets
        j_mask = j_offsets < N
        
        # Load A[i][j] and x[j]
        a_indices = i * N + j_offsets
        a_vals = tl.load(A_ptr + a_indices, mask=j_mask, other=0.0)
        x_vals = tl.load(x_ptr + j_offsets, mask=j_mask, other=0.0)
        
        # Accumulate tmp[i]
        tmp_val += tl.sum(a_vals * x_vals)
    
    # Store tmp[i]
    tl.store(tmp_ptr + i, tmp_val)
    
    # Synchronize to ensure all tmp values are computed
    tl.debug_barrier()
    
    # Load tmp[i]
    tmp_i = tl.load(tmp_ptr + i)
    
    # Update y[j] = y[j] + A[i][j] * tmp[i] for j in [0, N)
    for j_start in range(0, N, BLOCK_SIZE):
        j_offsets = j_start + offsets
        j_mask = j_offsets < N
        
        # Load A[i][j]
        a_indices = i * N + j_offsets
        a_vals = tl.load(A_ptr + a_indices, mask=j_mask, other=0.0)
        
        # Load current y[j]
        y_vals = tl.load(y_ptr + j_offsets, mask=j_mask, other=0.0)
        
        # Update y[j]
        new_y_vals = y_vals + a_vals * tmp_i
        tl.store(y_ptr + j_offsets, new_y_vals, mask=j_mask)

def atax_triton(A, tmp, x, y, M, N):
    BLOCK_SIZE = 64
    
    # Launch kernel with M programs (one per row of A)
    grid = (M,)
    
    atax_kernel[grid](A, tmp, x, y, M, N, BLOCK_SIZE)