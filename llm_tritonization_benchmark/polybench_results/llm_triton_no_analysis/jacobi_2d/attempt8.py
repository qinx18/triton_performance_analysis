import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_2d_kernel(A_ptr, B_ptr, N: tl.constexpr, TSTEPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    for t in range(TSTEPS):
        # First phase: compute B from A
        mask = offsets < (N-2) * (N-2)
        valid_offsets = tl.where(mask, offsets, 0)
        
        i_coords = 1 + valid_offsets // (N-2)
        j_coords = 1 + valid_offsets % (N-2)
        
        center_idx = i_coords * N + j_coords
        left_idx = i_coords * N + (j_coords - 1)
        right_idx = i_coords * N + (j_coords + 1)
        top_idx = (i_coords - 1) * N + j_coords
        bottom_idx = (i_coords + 1) * N + j_coords
        
        center = tl.load(A_ptr + center_idx, mask=mask)
        left = tl.load(A_ptr + left_idx, mask=mask)
        right = tl.load(A_ptr + right_idx, mask=mask)
        top = tl.load(A_ptr + top_idx, mask=mask)
        bottom = tl.load(A_ptr + bottom_idx, mask=mask)
        
        result_b = 0.2 * (center + left + right + bottom + top)
        tl.store(B_ptr + center_idx, result_b, mask=mask)
        
        # Second phase: compute A from B
        center = tl.load(B_ptr + center_idx, mask=mask)
        left = tl.load(B_ptr + left_idx, mask=mask)
        right = tl.load(B_ptr + right_idx, mask=mask)
        top = tl.load(B_ptr + top_idx, mask=mask)
        bottom = tl.load(B_ptr + bottom_idx, mask=mask)
        
        result_a = 0.2 * (center + left + right + bottom + top)
        tl.store(A_ptr + center_idx, result_a, mask=mask)

def jacobi_2d_triton(A, B, N, TSTEPS):
    BLOCK_SIZE = 256
    grid_size = triton.cdiv((N-2) * (N-2), BLOCK_SIZE)
    
    jacobi_2d_kernel[(grid_size,)](
        A, B, N, TSTEPS, BLOCK_SIZE
    )