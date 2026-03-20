import torch
import triton
import triton.language as tl

@triton.jit
def floyd_warshall_kernel(path_ptr, N: tl.constexpr, k: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program IDs for i and j dimensions
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    # Calculate starting positions
    i_start = pid_i * BLOCK_SIZE
    j_start = pid_j * BLOCK_SIZE
    
    # Create offset arrays
    i_offsets = tl.arange(0, BLOCK_SIZE)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Calculate actual indices
    i_indices = i_start + i_offsets
    j_indices = j_start + j_offsets
    
    # Create masks for bounds checking
    i_mask = i_indices < N
    j_mask = j_indices < N
    
    # Load path[i][j] values for the block
    path_ij_ptrs = path_ptr + i_indices[:, None] * N + j_indices[None, :]
    path_ij_mask = i_mask[:, None] & j_mask[None, :]
    path_ij = tl.load(path_ij_ptrs, mask=path_ij_mask)
    
    # Load path[i][k] values
    path_ik_ptrs = path_ptr + i_indices * N + k
    path_ik = tl.load(path_ik_ptrs, mask=i_mask)
    
    # Load path[k][j] values
    path_kj_ptrs = path_ptr + k * N + j_indices
    path_kj = tl.load(path_kj_ptrs, mask=j_mask)
    
    # Broadcast path_ik and path_kj for element-wise operations
    path_ik_broadcast = path_ik[:, None]
    path_kj_broadcast = path_kj[None, :]
    
    # Calculate path[i][k] + path[k][j]
    path_sum = path_ik_broadcast + path_kj_broadcast
    
    # Apply Floyd-Warshall update: take minimum
    path_ij_new = tl.minimum(path_ij, path_sum)
    
    # Store updated values back
    tl.store(path_ij_ptrs, path_ij_new, mask=path_ij_mask)

def floyd_warshall_triton(path, N):
    BLOCK_SIZE = 16
    
    # Calculate grid dimensions
    grid_i = triton.cdiv(N, BLOCK_SIZE)
    grid_j = triton.cdiv(N, BLOCK_SIZE)
    
    # Sequential loop over k (cannot be parallelized)
    for k in range(N):
        # Launch kernel for each k
        floyd_warshall_kernel[(grid_i, grid_j)](
            path,
            N,
            k,
            BLOCK_SIZE
        )