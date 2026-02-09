import triton
import triton.language as tl
import torch

@triton.jit
def gemver_kernel(A, u1, u2, v1, v2, w, x, y, z, alpha, beta, N, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate block start for i dimension
    block_start_i = pid * BLOCK_SIZE
    
    # Create offset vectors once
    offsets_i = tl.arange(0, BLOCK_SIZE)
    offsets_j = tl.arange(0, BLOCK_SIZE)
    
    # Current i offsets for this block
    current_i = block_start_i + offsets_i
    mask_i = current_i < N
    
    # Load u1, u2 for current i block
    u1_vals = tl.load(u1 + current_i, mask=mask_i, other=0.0)
    u2_vals = tl.load(u2 + current_i, mask=mask_i, other=0.0)
    
    # Loop 1: Update A matrix
    for j_start in range(0, N, BLOCK_SIZE):
        current_j = j_start + offsets_j
        mask_j = current_j < N
        
        # Load v1, v2 for current j block
        v1_vals = tl.load(v1 + current_j, mask=mask_j, other=0.0)
        v2_vals = tl.load(v2 + current_j, mask=mask_j, other=0.0)
        
        # Update A[i][j] for all combinations in this block
        i_vals_expanded = current_i[:, None]
        j_vals_expanded = current_j[None, :]
        mask_ij = mask_i[:, None] & mask_j[None, :]
        
        # Compute linear indices
        a_offsets = i_vals_expanded * N + j_vals_expanded
        
        # Load current A values
        a_vals = tl.load(A + a_offsets, mask=mask_ij, other=0.0)
        
        # Compute updates
        u1_expanded = u1_vals[:, None]
        u2_expanded = u2_vals[:, None]
        v1_expanded = v1_vals[None, :]
        v2_expanded = v2_vals[None, :]
        
        update = u1_expanded * v1_expanded + u2_expanded * v2_expanded
        
        # Update A
        new_a_vals = a_vals + update
        tl.store(A + a_offsets, new_a_vals, mask=mask_ij)
    
    # Initialize x accumulator
    x_accum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Loop 2: x[i] += beta * A[j][i] * y[j]
    for j in range(N):
        # Load y[j]
        y_val = tl.load(y + j)
        
        # Load A[j][current_i] (transpose access)
        a_offsets = j * N + current_i
        a_vals = tl.load(A + a_offsets, mask=mask_i, other=0.0)
        
        # Accumulate contribution
        x_accum += beta * a_vals * y_val
    
    # Load existing x values and add accumulated result plus z
    x_vals = tl.load(x + current_i, mask=mask_i, other=0.0)
    z_vals = tl.load(z + current_i, mask=mask_i, other=0.0)
    x_vals = x_vals + x_accum + z_vals
    
    # Store updated x
    tl.store(x + current_i, x_vals, mask=mask_i)

@triton.jit
def gemver_w_kernel(A, x, w, alpha, N, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate block start for i dimension
    block_start_i = pid * BLOCK_SIZE
    
    # Create offset vector once
    offsets_i = tl.arange(0, BLOCK_SIZE)
    
    # Current i offsets for this block
    current_i = block_start_i + offsets_i
    mask_i = current_i < N
    
    # Initialize w accumulator
    w_accum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Loop: w[i] += alpha * A[i][j] * x[j]
    for j in range(N):
        # Load x[j]
        x_val = tl.load(x + j)
        
        # Load A[current_i][j]
        a_offsets = current_i * N + j
        a_vals = tl.load(A + a_offsets, mask=mask_i, other=0.0)
        
        # Accumulate contribution
        w_accum += alpha * a_vals * x_val
    
    # Load existing w values and add accumulated result
    w_vals = tl.load(w + current_i, mask=mask_i, other=0.0)
    w_vals = w_vals + w_accum
    
    # Store final w
    tl.store(w + current_i, w_vals, mask=mask_i)

def gemver_triton(A, u1, u2, v1, v2, w, x, y, z, alpha, beta, N):
    BLOCK_SIZE = 32
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    gemver_kernel[grid](A, u1, u2, v1, v2, w, x, y, z, alpha, beta, N, BLOCK_SIZE)
    gemver_w_kernel[grid](A, x, w, alpha, N, BLOCK_SIZE)