import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_2d_kernel(A_ptr, B_ptr, N: tl.constexpr, TSTEPS: tl.constexpr, BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    i_start = pid_i * BLOCK_I + 1
    j_start = pid_j * BLOCK_J + 1
    
    i_offsets = tl.arange(0, BLOCK_I)
    j_offsets = tl.arange(0, BLOCK_J)
    
    for t in range(TSTEPS):
        # Synchronize between phases
        tl.debug_barrier()
        
        # First phase: update B from A
        i_indices = i_start + i_offsets
        j_indices = j_start + j_offsets[:, None]
        
        i_mask = i_indices < N - 1
        j_mask = j_indices < N - 1
        mask = i_mask[None, :] & j_mask
        
        # Load A values for stencil computation
        center_idx = i_indices[None, :] * N + j_indices
        left_idx = i_indices[None, :] * N + (j_indices - 1)
        right_idx = i_indices[None, :] * N + (j_indices + 1)
        up_idx = (i_indices[None, :] - 1) * N + j_indices
        down_idx = (i_indices[None, :] + 1) * N + j_indices
        
        a_center = tl.load(A_ptr + center_idx, mask=mask, other=0.0)
        a_left = tl.load(A_ptr + left_idx, mask=mask, other=0.0)
        a_right = tl.load(A_ptr + right_idx, mask=mask, other=0.0)
        a_up = tl.load(A_ptr + up_idx, mask=mask, other=0.0)
        a_down = tl.load(A_ptr + down_idx, mask=mask, other=0.0)
        
        b_new = 0.2 * (a_center + a_left + a_right + a_up + a_down)
        tl.store(B_ptr + center_idx, b_new, mask=mask)
        
        # Synchronize between phases
        tl.debug_barrier()
        
        # Second phase: update A from B
        b_center = tl.load(B_ptr + center_idx, mask=mask, other=0.0)
        b_left = tl.load(B_ptr + left_idx, mask=mask, other=0.0)
        b_right = tl.load(B_ptr + right_idx, mask=mask, other=0.0)
        b_up = tl.load(B_ptr + up_idx, mask=mask, other=0.0)
        b_down = tl.load(B_ptr + down_idx, mask=mask, other=0.0)
        
        a_new = 0.2 * (b_center + b_left + b_right + b_up + b_down)
        tl.store(A_ptr + center_idx, a_new, mask=mask)

def jacobi_2d_triton(A, B, N, TSTEPS):
    BLOCK_I = 16
    BLOCK_J = 16
    
    grid_i = triton.cdiv(N - 2, BLOCK_I)
    grid_j = triton.cdiv(N - 2, BLOCK_J)
    
    # Launch kernel for each timestep sequentially
    for t in range(TSTEPS):
        # Update B from A
        jacobi_2d_update_kernel[(grid_i, grid_j)](A, B, N, True, BLOCK_I, BLOCK_J)
        # Update A from B  
        jacobi_2d_update_kernel[(grid_i, grid_j)](B, A, N, False, BLOCK_I, BLOCK_J)

@triton.jit
def jacobi_2d_update_kernel(src_ptr, dst_ptr, N: tl.constexpr, is_first: tl.constexpr, BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    i_start = pid_i * BLOCK_I + 1
    j_start = pid_j * BLOCK_J + 1
    
    i_offsets = tl.arange(0, BLOCK_I)
    j_offsets = tl.arange(0, BLOCK_J)
    
    i_indices = i_start + i_offsets
    j_indices = j_start + j_offsets[:, None]
    
    i_mask = i_indices < N - 1
    j_mask = j_indices < N - 1
    mask = i_mask[None, :] & j_mask
    
    # Load source values for stencil computation
    center_idx = i_indices[None, :] * N + j_indices
    left_idx = i_indices[None, :] * N + (j_indices - 1)
    right_idx = i_indices[None, :] * N + (j_indices + 1)
    up_idx = (i_indices[None, :] - 1) * N + j_indices
    down_idx = (i_indices[None, :] + 1) * N + j_indices
    
    src_center = tl.load(src_ptr + center_idx, mask=mask, other=0.0)
    src_left = tl.load(src_ptr + left_idx, mask=mask, other=0.0)
    src_right = tl.load(src_ptr + right_idx, mask=mask, other=0.0)
    src_up = tl.load(src_ptr + up_idx, mask=mask, other=0.0)
    src_down = tl.load(src_ptr + down_idx, mask=mask, other=0.0)
    
    dst_new = 0.2 * (src_center + src_left + src_right + src_up + src_down)
    tl.store(dst_ptr + center_idx, dst_new, mask=mask)