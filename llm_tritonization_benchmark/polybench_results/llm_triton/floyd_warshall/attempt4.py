import triton
import triton.language as tl
import torch

@triton.jit
def floyd_warshall_kernel(path_ptr, path_copy_ptr, N, k, BLOCK_SIZE: tl.constexpr):
    # Get program ID for i dimension
    i_block = tl.program_id(0)
    
    # Compute i indices for this block
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_base = i_block * BLOCK_SIZE
    i = i_base + i_offsets
    i_mask = i < N
    
    # Process all j values in blocks
    j_offsets = tl.arange(0, BLOCK_SIZE)
    for j_block_start in range(0, N, BLOCK_SIZE):
        j = j_block_start + j_offsets
        j_mask = j < N
        
        # Create combined mask
        combined_mask = i_mask[:, None] & j_mask[None, :]
        
        # Compute linear indices for path[i][j], path[i][k], path[k][j]
        path_ij_idx = i[:, None] * N + j[None, :]
        path_ik_idx = i[:, None] * N + k
        path_kj_idx = k * N + j[None, :]
        
        # Load values with proper masking
        path_ij = tl.load(path_copy_ptr + path_ij_idx, mask=combined_mask)
        path_ik = tl.load(path_copy_ptr + path_ik_idx, mask=i_mask[:, None])
        path_kj = tl.load(path_copy_ptr + path_kj_idx, mask=j_mask[None, :])
        
        # Compute Floyd-Warshall update: min(path[i][j], path[i][k] + path[k][j])
        new_path = path_ik + path_kj
        result = tl.minimum(path_ij, new_path)
        
        # Store result only where mask is valid
        tl.store(path_ptr + path_ij_idx, result, mask=combined_mask)

def floyd_warshall_triton(path, N):
    BLOCK_SIZE = 16
    num_i_blocks = triton.cdiv(N, BLOCK_SIZE)
    
    # Sequential loop over k (cannot be parallelized due to dependencies)
    for k in range(N):
        # Create read-only copy to handle WAR dependencies
        path_copy = path.clone()
        
        # Launch kernel to parallelize over i dimension
        floyd_warshall_kernel[(num_i_blocks,)](
            path, path_copy, N, k, BLOCK_SIZE
        )