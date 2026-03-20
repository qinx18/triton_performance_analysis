import triton
import triton.language as tl
import torch

@triton.jit
def gemver_kernel(
    A_ptr, u1_ptr, u2_ptr, v1_ptr, v2_ptr, w_ptr, x_ptr, y_ptr, z_ptr,
    alpha, beta, N: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate which row this program handles
    row = pid
    
    if row >= N:
        return
    
    # First loop: A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j]
    if row < N:
        u1_val = tl.load(u1_ptr + row)
        u2_val = tl.load(u2_ptr + row)
        
        # Process columns in blocks
        col_offsets = tl.arange(0, BLOCK_SIZE)
        for col_start in range(0, N, BLOCK_SIZE):
            current_col_offsets = col_offsets + col_start
            col_mask = current_col_offsets < N
            
            # Load current A values
            a_offsets = row * N + current_col_offsets
            a_vals = tl.load(A_ptr + a_offsets, mask=col_mask)
            
            # Load v1 and v2 values
            v1_vals = tl.load(v1_ptr + current_col_offsets, mask=col_mask)
            v2_vals = tl.load(v2_ptr + current_col_offsets, mask=col_mask)
            
            # Update A values
            new_a_vals = a_vals + u1_val * v1_vals + u2_val * v2_vals
            
            # Store back
            tl.store(A_ptr + a_offsets, new_a_vals, mask=col_mask)

@triton.jit
def gemver_x_update_kernel(
    A_ptr, x_ptr, y_ptr, beta, N: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    # Second loop: x[i] = x[i] + beta * A[j][i] * y[j]
    pid = tl.program_id(0)
    col = pid
    
    if col >= N:
        return
    
    x_val = tl.load(x_ptr + col)
    
    # Sum over all rows for this column
    row_offsets = tl.arange(0, BLOCK_SIZE)
    for row_start in range(0, N, BLOCK_SIZE):
        current_row_offsets = row_offsets + row_start
        row_mask = current_row_offsets < N
        
        # Load A[row][col] values (transposed access)
        a_offsets = current_row_offsets * N + col
        a_vals = tl.load(A_ptr + a_offsets, mask=row_mask)
        
        # Load y values
        y_vals = tl.load(y_ptr + current_row_offsets, mask=row_mask)
        
        # Accumulate
        products = beta * a_vals * y_vals
        x_val += tl.sum(products)
    
    tl.store(x_ptr + col, x_val)

@triton.jit
def gemver_x_z_kernel(
    x_ptr, z_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    # Third loop: x[i] = x[i] + z[i]
    pid = tl.program_id(0)
    
    start_idx = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE) + start_idx
    mask = offsets < N
    
    x_vals = tl.load(x_ptr + offsets, mask=mask)
    z_vals = tl.load(z_ptr + offsets, mask=mask)
    
    result = x_vals + z_vals
    tl.store(x_ptr + offsets, result, mask=mask)

@triton.jit
def gemver_w_kernel(
    A_ptr, w_ptr, x_ptr, alpha, N: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    # Fourth loop: w[i] = w[i] + alpha * A[i][j] * x[j]
    pid = tl.program_id(0)
    row = pid
    
    if row >= N:
        return
    
    w_val = tl.load(w_ptr + row)
    
    # Sum over all columns for this row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    for col_start in range(0, N, BLOCK_SIZE):
        current_col_offsets = col_offsets + col_start
        col_mask = current_col_offsets < N
        
        # Load A[row][col] values
        a_offsets = row * N + current_col_offsets
        a_vals = tl.load(A_ptr + a_offsets, mask=col_mask)
        
        # Load x values
        x_vals = tl.load(x_ptr + current_col_offsets, mask=col_mask)
        
        # Accumulate
        products = alpha * a_vals * x_vals
        w_val += tl.sum(products)
    
    tl.store(w_ptr + row, w_val)

def gemver_triton(A, u1, u2, v1, v2, w, x, y, z, alpha, beta, N):
    BLOCK_SIZE = 64
    
    # First kernel: Update A matrix
    grid = (N,)
    gemver_kernel[grid](
        A, u1, u2, v1, v2, w, x, y, z,
        alpha, beta, N, BLOCK_SIZE
    )
    
    # Second kernel: Update x with A^T * y
    grid = (N,)
    gemver_x_update_kernel[grid](
        A, x, y, beta, N, BLOCK_SIZE
    )
    
    # Third kernel: Add z to x
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    gemver_x_z_kernel[grid](
        x, z, N, BLOCK_SIZE
    )
    
    # Fourth kernel: Update w
    grid = (N,)
    gemver_w_kernel[grid](
        A, w, x, alpha, N, BLOCK_SIZE
    )