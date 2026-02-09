import triton
import triton.language as tl
import torch

@triton.jit
def seidel_2d_kernel(A, N, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for t in range(40):  # TSTEPS = 40
        for block_start in range(0, (N - 2) * (N - 2), BLOCK_SIZE):
            indices = block_start + offsets
            mask = indices < (N - 2) * (N - 2)
            
            valid_indices = tl.where(mask, indices, 0)
            i = (valid_indices // (N - 2)) + 1
            j = (valid_indices % (N - 2)) + 1
            
            # Load all 9 neighbors
            idx_center = i * N + j
            idx_top_left = (i - 1) * N + (j - 1)
            idx_top = (i - 1) * N + j
            idx_top_right = (i - 1) * N + (j + 1)
            idx_left = i * N + (j - 1)
            idx_right = i * N + (j + 1)
            idx_bottom_left = (i + 1) * N + (j - 1)
            idx_bottom = (i + 1) * N + j
            idx_bottom_right = (i + 1) * N + (j + 1)
            
            val_center = tl.load(A + idx_center, mask=mask)
            val_top_left = tl.load(A + idx_top_left, mask=mask)
            val_top = tl.load(A + idx_top, mask=mask)
            val_top_right = tl.load(A + idx_top_right, mask=mask)
            val_left = tl.load(A + idx_left, mask=mask)
            val_right = tl.load(A + idx_right, mask=mask)
            val_bottom_left = tl.load(A + idx_bottom_left, mask=mask)
            val_bottom = tl.load(A + idx_bottom, mask=mask)
            val_bottom_right = tl.load(A + idx_bottom_right, mask=mask)
            
            # Compute average
            new_val = (val_top_left + val_top + val_top_right + 
                       val_left + val_center + val_right + 
                       val_bottom_left + val_bottom + val_bottom_right) / 9.0
            
            # Store result
            tl.store(A + idx_center, new_val, mask=mask)

def seidel_2d_triton(A, N, TSTEPS):
    BLOCK_SIZE = 256
    grid = (1,)
    
    seidel_2d_kernel[grid](A, N, BLOCK_SIZE)