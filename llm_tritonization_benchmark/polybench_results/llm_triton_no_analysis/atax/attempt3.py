import triton
import triton.language as tl
import torch

@triton.jit
def atax_kernel(A_ptr, tmp_ptr, x_ptr, y_ptr, M: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Initialize y to 0 (only first program does this)
    if pid == 0:
        y_offsets = tl.arange(0, BLOCK_SIZE)
        for block_start in range(0, N, BLOCK_SIZE):
            current_y_offsets = block_start + y_offsets
            y_mask = current_y_offsets < N
            tl.store(y_ptr + current_y_offsets, 0.0, mask=y_mask)
    
    tl.debug_barrier()
    
    # Each program handles one row of the outer loop
    i = pid
    if i >= M:
        return
    
    # First inner loop: tmp[i] = sum(A[i][j] * x[j])
    x_offsets = tl.arange(0, BLOCK_SIZE)
    tmp_val = 0.0
    
    for block_start in range(0, N, BLOCK_SIZE):
        current_x_offsets = block_start + x_offsets
        x_mask = current_x_offsets < N
        
        A_offsets = i * N + current_x_offsets
        A_vals = tl.load(A_ptr + A_offsets, mask=x_mask, other=0.0)
        x_vals = tl.load(x_ptr + current_x_offsets, mask=x_mask, other=0.0)
        
        tmp_val += tl.sum(A_vals * x_vals)
    
    tl.store(tmp_ptr + i, tmp_val)
    
    tl.debug_barrier()
    
    # Second inner loop: y[j] += A[i][j] * tmp[i]
    tmp_i = tl.load(tmp_ptr + i)
    y_offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, N, BLOCK_SIZE):
        current_y_offsets = block_start + y_offsets
        y_mask = current_y_offsets < N
        
        A_offsets = i * N + current_y_offsets
        A_vals = tl.load(A_ptr + A_offsets, mask=y_mask, other=0.0)
        
        # Use vectorized atomic add
        update_vals = A_vals * tmp_i
        tl.atomic_add(y_ptr + current_y_offsets, update_vals, mask=y_mask)

def atax_triton(A, tmp, x, y, M, N):
    BLOCK_SIZE = 64
    
    grid = (M,)
    
    atax_kernel[grid](
        A, tmp, x, y,
        M=M, N=N, BLOCK_SIZE=BLOCK_SIZE
    )