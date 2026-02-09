import triton
import triton.language as tl
import torch

@triton.jit
def floyd_warshall_kernel(path_ptr, path_copy_ptr, N, k, BLOCK_SIZE: tl.constexpr):
    # Get program ID for i dimension
    i_block = tl.program_id(0)
    j_block = tl.program_id(1)
    
    # Compute base offsets
    i_offsets = tl.arange(0, BLOCK_SIZE)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Compute actual indices
    i = i_block * BLOCK_SIZE + i_offsets
    j = j_block * BLOCK_SIZE + j_offsets
    
    # Create masks
    i_mask = i < N
    j_mask = j < N
    
    # Compute linear indices
    path_ij_idx = i[:, None] * N + j[None, :]
    path_ik_idx = i[:, None] * N + k
    path_kj_idx = k * N + j[None, :]
    
    # Create combined mask
    mask = i_mask[:, None] & j_mask[None, :]
    
    # Load values
    path_ij = tl.load(path_copy_ptr + path_ij_idx, mask=mask, other=float('inf'))
    path_ik = tl.load(path_copy_ptr + path_ik_idx, mask=i_mask[:, None], other=float('inf'))
    path_kj = tl.load(path_copy_ptr + path_kj_idx, mask=j_mask[None, :], other=float('inf'))
    
    # Compute Floyd-Warshall update
    new_path = path_ik + path_kj
    result = tl.minimum(path_ij, new_path)
    
    # Store result
    tl.store(path_ptr + path_ij_idx, result, mask=mask)

def floyd_warshall_triton(path, N):
    BLOCK_SIZE = 16
    num_i_blocks = triton.cdiv(N, BLOCK_SIZE)
    num_j_blocks = triton.cdiv(N, BLOCK_SIZE)
    
    for k in range(N):
        path_copy = path.clone()
        
        floyd_warshall_kernel[(num_i_blocks, num_j_blocks)](
            path, path_copy, N, k, BLOCK_SIZE
        )