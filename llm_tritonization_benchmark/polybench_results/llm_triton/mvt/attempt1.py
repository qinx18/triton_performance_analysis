import triton
import triton.language as tl
import torch

@triton.jit
def mvt_kernel(A_ptr, x1_ptr, x2_ptr, y_1_ptr, y_2_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Get program ID for the current block
    pid = tl.program_id(axis=0)
    
    # Calculate which row this block will process
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    i_mask = i_offsets < N
    
    # Initialize accumulator for x1
    x1_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Load current x1 values
    x1_vals = tl.load(x1_ptr + i_offsets, mask=i_mask, other=0.0)
    x1_acc = x1_vals
    
    # First loop: x1[i] = x1[i] + A[i][j] * y_1[j]
    j_offsets = tl.arange(0, BLOCK_SIZE)
    for j_start in range(0, N, BLOCK_SIZE):
        j_current = j_start + j_offsets
        j_mask = j_current < N
        
        # Load y_1[j] values
        y_1_vals = tl.load(y_1_ptr + j_current, mask=j_mask, other=0.0)
        
        # For each i in this block, accumulate A[i][j] * y_1[j]
        for i_idx in range(BLOCK_SIZE):
            if block_start + i_idx < N:
                # Load A[i][j] values for this row
                i_val = block_start + i_idx
                A_row_offsets = i_val * N + j_current
                A_vals = tl.load(A_ptr + A_row_offsets, mask=j_mask, other=0.0)
                
                # Compute dot product contribution
                contribution = tl.sum(A_vals * y_1_vals)
                x1_acc = tl.where(offsets == i_idx, x1_acc[i_idx] + contribution, x1_acc)
    
    # Store updated x1 values
    tl.store(x1_ptr + i_offsets, x1_acc, mask=i_mask)
    
    # Initialize accumulator for x2
    x2_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Load current x2 values
    x2_vals = tl.load(x2_ptr + i_offsets, mask=i_mask, other=0.0)
    x2_acc = x2_vals
    
    # Second loop: x2[i] = x2[i] + A[j][i] * y_2[j]
    for j_start in range(0, N, BLOCK_SIZE):
        j_current = j_start + j_offsets
        j_mask = j_current < N
        
        # Load y_2[j] values
        y_2_vals = tl.load(y_2_ptr + j_current, mask=j_mask, other=0.0)
        
        # For each i in this block, accumulate A[j][i] * y_2[j]
        for i_idx in range(BLOCK_SIZE):
            if block_start + i_idx < N:
                i_val = block_start + i_idx
                # A[j][i] means row j, column i
                A_col_offsets = j_current * N + i_val
                A_vals = tl.load(A_ptr + A_col_offsets, mask=j_mask, other=0.0)
                
                # Compute dot product contribution
                contribution = tl.sum(A_vals * y_2_vals)
                x2_acc = tl.where(offsets == i_idx, x2_acc[i_idx] + contribution, x2_acc)
    
    # Store updated x2 values
    tl.store(x2_ptr + i_offsets, x2_acc, mask=i_mask)


def mvt_triton(A, x1, x2, y_1, y_2, N):
    BLOCK_SIZE = 128
    
    # Ensure tensors are contiguous and on GPU
    A = A.contiguous()
    x1 = x1.contiguous()
    x2 = x2.contiguous()
    y_1 = y_1.contiguous()
    y_2 = y_2.contiguous()
    
    # Calculate grid size
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    # Launch kernel
    mvt_kernel[grid](
        A, x1, x2, y_1, y_2, N,
        BLOCK_SIZE=BLOCK_SIZE
    )