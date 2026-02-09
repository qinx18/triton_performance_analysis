import torch
import triton
import triton.language as tl

@triton.jit
def atax_kernel(A_ptr, tmp_ptr, x_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Phase 1: Initialize y to zero (done by first program)
    if pid == 0:
        offsets = tl.arange(0, BLOCK_SIZE)
        for start in range(0, N, BLOCK_SIZE):
            current_offsets = start + offsets
            mask = current_offsets < N
            tl.store(y_ptr + current_offsets, 0.0, mask=mask)
    
    # Phase 2: Each program processes one row i
    i = pid
    if i >= M:
        return
    
    # Initialize tmp[i] = 0.0
    tmp_val = 0.0
    
    # Compute tmp[i] = sum(A[i][j] * x[j]) for all j
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

@triton.jit
def atax_kernel_phase2(A_ptr, tmp_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i = pid
    
    if i >= M:
        return
    
    # Load tmp[i]
    tmp_i = tl.load(tmp_ptr + i)
    
    # Update y[j] += A[i][j] * tmp[i] for all j
    offsets = tl.arange(0, BLOCK_SIZE)
    for j_start in range(0, N, BLOCK_SIZE):
        j_offsets = j_start + offsets
        j_mask = j_offsets < N
        
        # Load A[i][j]
        a_indices = i * N + j_offsets
        a_vals = tl.load(A_ptr + a_indices, mask=j_mask, other=0.0)
        
        # Atomically update y[j]
        updates = a_vals * tmp_i
        tl.atomic_add(y_ptr + j_offsets, updates, mask=j_mask)

def atax_triton(A, tmp, x, y, M, N):
    BLOCK_SIZE = 64
    grid = (M,)
    
    # Phase 1: Compute tmp values
    atax_kernel[grid](A, tmp, x, y, M, N, BLOCK_SIZE)
    
    # Phase 2: Update y values
    atax_kernel_phase2[grid](A, tmp, y, M, N, BLOCK_SIZE)