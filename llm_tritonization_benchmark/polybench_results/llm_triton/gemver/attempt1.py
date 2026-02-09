import triton
import triton.language as tl
import torch

@triton.jit
def gemver_kernel(A_ptr, u1_ptr, u2_ptr, v1_ptr, v2_ptr, w_ptr, x_ptr, y_ptr, z_ptr, 
                  alpha, beta, N, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate offsets
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # First loop: Update A matrix
    for i in range(0, N, BLOCK_SIZE):
        i_offsets = i + offsets
        i_mask = i_offsets < N
        
        # Load u1[i], u2[i]
        u1_vals = tl.load(u1_ptr + i_offsets, mask=i_mask, other=0.0)
        u2_vals = tl.load(u2_ptr + i_offsets, mask=i_mask, other=0.0)
        
        for j in range(0, N, BLOCK_SIZE):
            j_offsets = j + offsets
            j_mask = j_offsets < N
            
            # Load v1[j], v2[j]
            v1_vals = tl.load(v1_ptr + j_offsets, mask=j_mask, other=0.0)
            v2_vals = tl.load(v2_ptr + j_offsets, mask=j_mask, other=0.0)
            
            # Update A[i][j] for all combinations
            for ii in range(BLOCK_SIZE):
                if i + ii < N:
                    row_idx = i + ii
                    A_row_ptr = A_ptr + row_idx * N
                    
                    # Load A[i][j] values
                    A_vals = tl.load(A_row_ptr + j_offsets, mask=j_mask, other=0.0)
                    
                    # Update: A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j]
                    u1_i = tl.load(u1_ptr + row_idx)
                    u2_i = tl.load(u2_ptr + row_idx)
                    
                    A_vals = A_vals + u1_i * v1_vals + u2_i * v2_vals
                    
                    # Store back
                    tl.store(A_row_ptr + j_offsets, A_vals, mask=j_mask)
    
    # Second loop: Update x with beta * A^T * y
    for i in range(0, N, BLOCK_SIZE):
        i_offsets = i + offsets
        i_mask = i_offsets < N
        
        # Load x[i]
        x_vals = tl.load(x_ptr + i_offsets, mask=i_mask, other=0.0)
        
        for j in range(0, N, BLOCK_SIZE):
            j_offsets = j + offsets
            j_mask = j_offsets < N
            
            # Load y[j]
            y_vals = tl.load(y_ptr + j_offsets, mask=j_mask, other=0.0)
            
            # Update x[i] += beta * A[j][i] * y[j]
            for jj in range(BLOCK_SIZE):
                if j + jj < N:
                    j_idx = j + jj
                    y_j = tl.load(y_ptr + j_idx)
                    
                    # Load A[j][i] - transpose access
                    A_ji_ptr = A_ptr + j_idx * N
                    A_ji_vals = tl.load(A_ji_ptr + i_offsets, mask=i_mask, other=0.0)
                    
                    x_vals = x_vals + beta * A_ji_vals * y_j
        
        # Store x[i]
        tl.store(x_ptr + i_offsets, x_vals, mask=i_mask)
    
    # Third loop: x[i] += z[i]
    for i in range(0, N, BLOCK_SIZE):
        i_offsets = i + offsets
        i_mask = i_offsets < N
        
        x_vals = tl.load(x_ptr + i_offsets, mask=i_mask, other=0.0)
        z_vals = tl.load(z_ptr + i_offsets, mask=i_mask, other=0.0)
        
        x_vals = x_vals + z_vals
        
        tl.store(x_ptr + i_offsets, x_vals, mask=i_mask)
    
    # Fourth loop: Update w with alpha * A * x
    for i in range(0, N, BLOCK_SIZE):
        i_offsets = i + offsets
        i_mask = i_offsets < N
        
        # Load w[i]
        w_vals = tl.load(w_ptr + i_offsets, mask=i_mask, other=0.0)
        
        for j in range(0, N, BLOCK_SIZE):
            j_offsets = j + offsets
            j_mask = j_offsets < N
            
            # Load x[j]
            x_vals = tl.load(x_ptr + j_offsets, mask=j_mask, other=0.0)
            
            # Update w[i] += alpha * A[i][j] * x[j]
            for ii in range(BLOCK_SIZE):
                if i + ii < N:
                    row_idx = i + ii
                    A_row_ptr = A_ptr + row_idx * N
                    
                    # Load A[i][j]
                    A_vals = tl.load(A_row_ptr + j_offsets, mask=j_mask, other=0.0)
                    
                    # Compute contribution to w[i]
                    contrib = tl.sum(alpha * A_vals * x_vals)
                    
                    # Load current w[i] and update
                    w_i = tl.load(w_ptr + row_idx)
                    w_i = w_i + contrib
                    tl.store(w_ptr + row_idx, w_i)

def gemver_triton(A, u1, u2, v1, v2, w, x, y, z, alpha, beta, N):
    BLOCK_SIZE = 32
    
    # Launch kernel with single program
    grid = (1,)
    
    gemver_kernel[grid](
        A, u1, u2, v1, v2, w, x, y, z,
        alpha, beta, N,
        BLOCK_SIZE=BLOCK_SIZE
    )