import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_2d_kernel(A_ptr, B_ptr, N, TSTEPS, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for t in range(TSTEPS):
        # First loop: compute B from A
        for block_start in range(0, (N-2)*(N-2), BLOCK_SIZE):
            current_offsets = block_start + offsets
            mask = current_offsets < (N-2)*(N-2)
            
            # Convert linear index to 2D coordinates for interior points
            linear_idx = current_offsets
            i = (linear_idx // (N-2)) + 1
            j = (linear_idx % (N-2)) + 1
            
            # Check bounds
            valid_mask = mask & (i >= 1) & (i < N-1) & (j >= 1) & (j < N-1)
            
            # Load A values with 5-point stencil
            center_idx = i * N + j
            left_idx = i * N + (j - 1)
            right_idx = i * N + (j + 1)
            up_idx = (i - 1) * N + j
            down_idx = (i + 1) * N + j
            
            a_center = tl.load(A_ptr + center_idx, mask=valid_mask, other=0.0)
            a_left = tl.load(A_ptr + left_idx, mask=valid_mask, other=0.0)
            a_right = tl.load(A_ptr + right_idx, mask=valid_mask, other=0.0)
            a_up = tl.load(A_ptr + up_idx, mask=valid_mask, other=0.0)
            a_down = tl.load(A_ptr + down_idx, mask=valid_mask, other=0.0)
            
            # Compute B[i][j]
            b_val = 0.2 * (a_center + a_left + a_right + a_up + a_down)
            
            # Store to B
            tl.store(B_ptr + center_idx, b_val, mask=valid_mask)
        
        # Second loop: compute A from B
        for block_start in range(0, (N-2)*(N-2), BLOCK_SIZE):
            current_offsets = block_start + offsets
            mask = current_offsets < (N-2)*(N-2)
            
            # Convert linear index to 2D coordinates for interior points
            linear_idx = current_offsets
            i = (linear_idx // (N-2)) + 1
            j = (linear_idx % (N-2)) + 1
            
            # Check bounds
            valid_mask = mask & (i >= 1) & (i < N-1) & (j >= 1) & (j < N-1)
            
            # Load B values with 5-point stencil
            center_idx = i * N + j
            left_idx = i * N + (j - 1)
            right_idx = i * N + (j + 1)
            up_idx = (i - 1) * N + j
            down_idx = (i + 1) * N + j
            
            b_center = tl.load(B_ptr + center_idx, mask=valid_mask, other=0.0)
            b_left = tl.load(B_ptr + left_idx, mask=valid_mask, other=0.0)
            b_right = tl.load(B_ptr + right_idx, mask=valid_mask, other=0.0)
            b_up = tl.load(B_ptr + up_idx, mask=valid_mask, other=0.0)
            b_down = tl.load(B_ptr + down_idx, mask=valid_mask, other=0.0)
            
            # Compute A[i][j]
            a_val = 0.2 * (b_center + b_left + b_right + b_up + b_down)
            
            # Store to A
            tl.store(A_ptr + center_idx, a_val, mask=valid_mask)

def jacobi_2d_triton(A, B, N, TSTEPS):
    BLOCK_SIZE = 256
    
    jacobi_2d_kernel[(1,)](
        A, B, N, TSTEPS, BLOCK_SIZE
    )