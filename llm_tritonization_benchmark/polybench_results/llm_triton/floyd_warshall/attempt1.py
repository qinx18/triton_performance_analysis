import triton
import triton.language as tl
import torch

@triton.jit
def floyd_warshall_kernel(path_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Get block indices
    block_i = tl.program_id(0)
    block_j = tl.program_id(1)
    
    # Compute base offsets once
    i_offsets = tl.arange(0, BLOCK_SIZE)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Compute actual indices
    i_indices = block_i * BLOCK_SIZE + i_offsets
    j_indices = block_j * BLOCK_SIZE + j_offsets
    
    # Create masks
    i_mask = i_indices < N
    j_mask = j_indices < N
    
    # Sequential k loop (cannot be parallelized due to dependencies)
    for k in range(N):
        # Load path[i][j] values
        ij_offsets = i_indices[:, None] * N + j_indices[None, :]
        ij_mask = i_mask[:, None] & j_mask[None, :]
        path_ij = tl.load(path_ptr + ij_offsets, mask=ij_mask)
        
        # Load path[i][k] values
        ik_offsets = i_indices * N + k
        ik_mask = i_mask & (k < N)
        path_ik = tl.load(path_ptr + ik_offsets, mask=ik_mask)
        
        # Load path[k][j] values
        kj_offsets = k * N + j_indices
        kj_mask = j_mask & (k < N)
        path_kj = tl.load(path_ptr + kj_offsets, mask=kj_mask)
        
        # Compute new distances
        new_dist = path_ik[:, None] + path_kj[None, :]
        
        # Update path[i][j] with minimum
        updated_path = tl.minimum(path_ij, new_dist)
        
        # Store back
        tl.store(path_ptr + ij_offsets, updated_path, mask=ij_mask)

def floyd_warshall_triton(path, N):
    BLOCK_SIZE = 16
    
    # Calculate grid dimensions
    grid_dim = triton.cdiv(N, BLOCK_SIZE)
    grid = (grid_dim, grid_dim)
    
    # Launch kernel
    floyd_warshall_kernel[grid](
        path, N, BLOCK_SIZE
    )