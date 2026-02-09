import triton
import triton.language as tl
import torch

@triton.jit
def mvt_kernel(A_ptr, x1_ptr, x2_ptr, y_1_ptr, y_2_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    mask_i = i_offsets < N
    
    # Load initial values for x1 and x2
    x1_vals = tl.load(x1_ptr + i_offsets, mask=mask_i, other=0.0)
    x2_vals = tl.load(x2_ptr + i_offsets, mask=mask_i, other=0.0)
    
    # First loop: x1[i] += A[i][j] * y_1[j]
    j_offsets = tl.arange(0, BLOCK_SIZE)
    for j_start in range(0, N, BLOCK_SIZE):
        j_current = j_start + j_offsets
        mask_j = j_current < N
        
        # Load y_1[j]
        y_1_vals = tl.load(y_1_ptr + j_current, mask=mask_j, other=0.0)
        
        # Compute A[i][j] indices for all i,j combinations
        i_expanded = i_offsets[:, None]
        j_expanded = j_current[None, :]
        A_indices = i_expanded * N + j_expanded
        
        mask_ij = mask_i[:, None] & mask_j[None, :]
        
        # Load A[i][j] values
        A_vals = tl.load(A_ptr + A_indices, mask=mask_ij, other=0.0)
        
        # Compute A[i][j] * y_1[j] and sum over j
        products = A_vals * y_1_vals[None, :]
        x1_vals += tl.sum(products, axis=1)
    
    # Second loop: x2[i] += A[j][i] * y_2[j]
    for j_start in range(0, N, BLOCK_SIZE):
        j_current = j_start + j_offsets
        mask_j = j_current < N
        
        # Load y_2[j]
        y_2_vals = tl.load(y_2_ptr + j_current, mask=mask_j, other=0.0)
        
        # Compute A[j][i] indices for all i,j combinations
        j_expanded = j_current[:, None]
        i_expanded = i_offsets[None, :]
        A_indices = j_expanded * N + i_expanded
        
        mask_ji = mask_j[:, None] & mask_i[None, :]
        
        # Load A[j][i] values
        A_vals = tl.load(A_ptr + A_indices, mask=mask_ji, other=0.0)
        
        # Compute A[j][i] * y_2[j] and sum over j
        products = A_vals * y_2_vals[:, None]
        x2_vals += tl.sum(products, axis=0)
    
    # Store results
    tl.store(x1_ptr + i_offsets, x1_vals, mask=mask_i)
    tl.store(x2_ptr + i_offsets, x2_vals, mask=mask_i)

def mvt_triton(A, x1, x2, y_1, y_2, N):
    BLOCK_SIZE = 32
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    mvt_kernel[grid](A, x1, x2, y_1, y_2, N, BLOCK_SIZE)