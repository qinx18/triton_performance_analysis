import triton
import triton.language as tl
import torch

@triton.jit
def gemver_phase1_kernel(A_ptr, u1_ptr, u2_ptr, v1_ptr, v2_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Phase 1: A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j]
    row = tl.program_id(0)
    
    # Load u1[row] and u2[row] once per row
    u1_val = tl.load(u1_ptr + row)
    u2_val = tl.load(u2_ptr + row)
    
    # Process columns in blocks
    col_offsets = tl.arange(0, BLOCK_SIZE)
    for col_start in range(0, N, BLOCK_SIZE):
        cols = col_start + col_offsets
        mask = cols < N
        
        # Load vectors
        v1_vals = tl.load(v1_ptr + cols, mask=mask)
        v2_vals = tl.load(v2_ptr + cols, mask=mask)
        
        # Load A[row, cols]
        a_idx = row * N + cols
        a_vals = tl.load(A_ptr + a_idx, mask=mask)
        
        # Compute update
        update = u1_val * v1_vals + u2_val * v2_vals
        new_a_vals = a_vals + update
        
        # Store back
        tl.store(A_ptr + a_idx, new_a_vals, mask=mask)

@triton.jit
def gemver_phase2_kernel(A_ptr, x_ptr, y_ptr, beta, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Phase 2: x[i] = x[i] + beta * A[j][i] * y[j] (sum over j)
    col = tl.program_id(0)
    
    # Load current x[col]
    x_val = tl.load(x_ptr + col)
    
    # Process rows in blocks and accumulate
    row_offsets = tl.arange(0, BLOCK_SIZE)
    acc = 0.0
    
    for row_start in range(0, N, BLOCK_SIZE):
        rows = row_start + row_offsets
        mask = rows < N
        
        # Load A[rows, col] and y[rows]
        a_idx = rows * N + col
        a_vals = tl.load(A_ptr + a_idx, mask=mask, other=0.0)
        y_vals = tl.load(y_ptr + rows, mask=mask, other=0.0)
        
        # Accumulate beta * A[j][col] * y[j]
        acc += tl.sum(beta * a_vals * y_vals)
    
    # Update x[col]
    new_x_val = x_val + acc
    tl.store(x_ptr + col, new_x_val)

@triton.jit
def gemver_phase3_kernel(x_ptr, z_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Phase 3: x[i] = x[i] + z[i]
    pid = tl.program_id(0)
    
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    x_vals = tl.load(x_ptr + offsets, mask=mask)
    z_vals = tl.load(z_ptr + offsets, mask=mask)
    
    new_x_vals = x_vals + z_vals
    tl.store(x_ptr + offsets, new_x_vals, mask=mask)

@triton.jit
def gemver_phase4_kernel(A_ptr, w_ptr, x_ptr, alpha, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Phase 4: w[i] = w[i] + alpha * A[i][j] * x[j] (sum over j)
    row = tl.program_id(0)
    
    # Load current w[row]
    w_val = tl.load(w_ptr + row)
    
    # Process columns in blocks and accumulate
    col_offsets = tl.arange(0, BLOCK_SIZE)
    acc = 0.0
    
    for col_start in range(0, N, BLOCK_SIZE):
        cols = col_start + col_offsets
        mask = cols < N
        
        # Load A[row, cols] and x[cols]
        a_idx = row * N + cols
        a_vals = tl.load(A_ptr + a_idx, mask=mask, other=0.0)
        x_vals = tl.load(x_ptr + cols, mask=mask, other=0.0)
        
        # Accumulate alpha * A[row][j] * x[j]
        acc += tl.sum(alpha * a_vals * x_vals)
    
    # Update w[row]
    new_w_val = w_val + acc
    tl.store(w_ptr + row, new_w_val)

def gemver_triton(A, u1, u2, v1, v2, w, x, y, z, alpha, beta, N):
    BLOCK_SIZE = 64
    
    # Phase 1: Update matrix A
    grid1 = (N,)
    gemver_phase1_kernel[grid1](A, u1, u2, v1, v2, N, BLOCK_SIZE)
    
    # Phase 2: Update x with column reduction
    grid2 = (N,)
    gemver_phase2_kernel[grid2](A, x, y, beta, N, BLOCK_SIZE)
    
    # Phase 3: Add z to x (fused with phase 2 conceptually, but separate for clarity)
    grid3 = (triton.cdiv(N, BLOCK_SIZE),)
    gemver_phase3_kernel[grid3](x, z, N, BLOCK_SIZE)
    
    # Phase 4: Update w with row reduction
    grid4 = (N,)
    gemver_phase4_kernel[grid4](A, w, x, alpha, N, BLOCK_SIZE)