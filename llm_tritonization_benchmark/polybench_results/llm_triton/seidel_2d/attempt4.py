import triton
import triton.language as tl
import torch

@triton.jit
def seidel_2d_kernel(A, N, TSTEPS, BLOCK_SIZE: tl.constexpr):
    for t in range(TSTEPS):
        for i in range(1, N - 1):
            offsets = tl.arange(0, BLOCK_SIZE)
            for block_start in range(1, N - 1, BLOCK_SIZE):
                j_indices = block_start + offsets
                mask = j_indices < N - 1
                
                # Calculate linear indices for 9-point stencil
                idx_center = i * N + j_indices
                idx_top_left = (i - 1) * N + (j_indices - 1)
                idx_top = (i - 1) * N + j_indices
                idx_top_right = (i - 1) * N + (j_indices + 1)
                idx_left = i * N + (j_indices - 1)
                idx_right = i * N + (j_indices + 1)
                idx_bottom_left = (i + 1) * N + (j_indices - 1)
                idx_bottom = (i + 1) * N + j_indices
                idx_bottom_right = (i + 1) * N + (j_indices + 1)
                
                # Load values
                val_top_left = tl.load(A + idx_top_left, mask=mask)
                val_top = tl.load(A + idx_top, mask=mask)
                val_top_right = tl.load(A + idx_top_right, mask=mask)
                val_left = tl.load(A + idx_left, mask=mask)
                val_center = tl.load(A + idx_center, mask=mask)
                val_right = tl.load(A + idx_right, mask=mask)
                val_bottom_left = tl.load(A + idx_bottom_left, mask=mask)
                val_bottom = tl.load(A + idx_bottom, mask=mask)
                val_bottom_right = tl.load(A + idx_bottom_right, mask=mask)
                
                # Compute 9-point average
                new_val = (val_top_left + val_top + val_top_right +
                          val_left + val_center + val_right +
                          val_bottom_left + val_bottom + val_bottom_right) / 9.0
                
                # Store result
                tl.store(A + idx_center, new_val, mask=mask)

def seidel_2d_triton(A, N, TSTEPS):
    BLOCK_SIZE = 256
    grid = (1,)
    
    seidel_2d_kernel[grid](A, N, TSTEPS, BLOCK_SIZE)