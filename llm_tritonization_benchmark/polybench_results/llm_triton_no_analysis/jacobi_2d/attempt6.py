import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_2d_kernel(A_ptr, B_ptr, N: tl.constexpr, TSTEPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for t in range(TSTEPS):
        # First phase: compute B from A
        for block_idx in range(block_start, block_start + BLOCK_SIZE):
            if block_idx >= (N-2) * (N-2):
                break
            
            i = 1 + block_idx // (N-2)
            j = 1 + block_idx % (N-2)
            
            if i < N-1 and j < N-1:
                center = tl.load(A_ptr + i * N + j)
                left = tl.load(A_ptr + i * N + (j-1))
                right = tl.load(A_ptr + i * N + (j+1))
                top = tl.load(A_ptr + (i-1) * N + j)
                bottom = tl.load(A_ptr + (i+1) * N + j)
                
                result = 0.2 * (center + left + right + bottom + top)
                tl.store(B_ptr + i * N + j, result)
        
        # Second phase: compute A from B
        for block_idx in range(block_start, block_start + BLOCK_SIZE):
            if block_idx >= (N-2) * (N-2):
                break
            
            i = 1 + block_idx // (N-2)
            j = 1 + block_idx % (N-2)
            
            if i < N-1 and j < N-1:
                center = tl.load(B_ptr + i * N + j)
                left = tl.load(B_ptr + i * N + (j-1))
                right = tl.load(B_ptr + i * N + (j+1))
                top = tl.load(B_ptr + (i-1) * N + j)
                bottom = tl.load(B_ptr + (i+1) * N + j)
                
                result = 0.2 * (center + left + right + bottom + top)
                tl.store(A_ptr + i * N + j, result)

def jacobi_2d_triton(A, B, N, TSTEPS):
    BLOCK_SIZE = 256
    grid_size = triton.cdiv((N-2) * (N-2), BLOCK_SIZE)
    
    jacobi_2d_kernel[(grid_size,)](
        A, B, N, TSTEPS, BLOCK_SIZE
    )