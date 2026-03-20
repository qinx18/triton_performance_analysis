import triton
import triton.language as tl
import torch

@triton.jit
def floyd_warshall_kernel(path_ptr, N: tl.constexpr, BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr, k: tl.constexpr):
    # Get program IDs for i and j dimensions
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    # Compute starting indices for this block
    i_start = pid_i * BLOCK_I
    j_start = pid_j * BLOCK_J
    
    # Create offset arrays
    i_offsets = i_start + tl.arange(0, BLOCK_I)
    j_offsets = j_start + tl.arange(0, BLOCK_J)
    
    # Create masks for valid indices
    i_mask = i_offsets < N
    j_mask = j_offsets < N
    
    # Create 2D masks
    mask = i_mask[:, None] & j_mask[None, :]
    
    # Compute linear indices for path[i][j]
    path_ij_offsets = i_offsets[:, None] * N + j_offsets[None, :]
    
    # Load current path[i][j] values
    path_ij = tl.load(path_ptr + path_ij_offsets, mask=mask, other=0.0)
    
    # Load path[i][k] values (broadcast across j dimension)
    path_ik_offsets = i_offsets * N + k
    i_mask_k = i_offsets < N
    path_ik = tl.load(path_ptr + path_ik_offsets, mask=i_mask_k, other=0.0)
    path_ik_broadcast = path_ik[:, None]
    
    # Load path[k][j] values (broadcast across i dimension)
    path_kj_offsets = k * N + j_offsets
    j_mask_k = j_offsets < N
    path_kj = tl.load(path_ptr + path_kj_offsets, mask=j_mask_k, other=0.0)
    path_kj_broadcast = path_kj[None, :]
    
    # Compute path[i][k] + path[k][j]
    path_sum = path_ik_broadcast + path_kj_broadcast
    
    # Update path[i][j] = min(path[i][j], path[i][k] + path[k][j])
    new_path_ij = tl.minimum(path_ij, path_sum)
    
    # Store the result
    tl.store(path_ptr + path_ij_offsets, new_path_ij, mask=mask)

def floyd_warshall_triton(path, N):
    # Ensure path is contiguous
    path = path.contiguous()
    
    # Choose block sizes
    BLOCK_I = min(triton.next_power_of_2(N), 32)
    BLOCK_J = min(triton.next_power_of_2(N), 32)
    
    # Compute grid dimensions
    grid_i = triton.cdiv(N, BLOCK_I)
    grid_j = triton.cdiv(N, BLOCK_J)
    
    # Sequential k loop with kernel launches
    for k in range(N):
        floyd_warshall_kernel[(grid_i, grid_j)](
            path,
            N=N,
            BLOCK_I=BLOCK_I,
            BLOCK_J=BLOCK_J,
            k=k
        )