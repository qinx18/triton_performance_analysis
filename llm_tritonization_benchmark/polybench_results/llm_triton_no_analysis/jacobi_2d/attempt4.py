import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_2d_kernel(A_ptr, B_ptr, N: tl.constexpr, TSTEPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    base_i = pid_i * BLOCK_SIZE + 1
    base_j = pid_j * BLOCK_SIZE + 1
    
    offsets_i = tl.arange(0, BLOCK_SIZE)
    offsets_j = tl.arange(0, BLOCK_SIZE)
    
    for t in range(TSTEPS):
        # First phase: update B from A
        for block_i in range(BLOCK_SIZE):
            i = base_i + block_i
            if i < N - 1:
                for block_j in range(BLOCK_SIZE):
                    j = base_j + block_j
                    if j < N - 1:
                        center = tl.load(A_ptr + i * N + j)
                        left = tl.load(A_ptr + i * N + (j - 1))
                        right = tl.load(A_ptr + i * N + (j + 1))
                        up = tl.load(A_ptr + (i - 1) * N + j)
                        down = tl.load(A_ptr + (i + 1) * N + j)
                        
                        result = 0.2 * (center + left + right + up + down)
                        tl.store(B_ptr + i * N + j, result)
        
        # Second phase: update A from B
        for block_i in range(BLOCK_SIZE):
            i = base_i + block_i
            if i < N - 1:
                for block_j in range(BLOCK_SIZE):
                    j = base_j + block_j
                    if j < N - 1:
                        center = tl.load(B_ptr + i * N + j)
                        left = tl.load(B_ptr + i * N + (j - 1))
                        right = tl.load(B_ptr + i * N + (j + 1))
                        up = tl.load(B_ptr + (i - 1) * N + j)
                        down = tl.load(B_ptr + (i + 1) * N + j)
                        
                        result = 0.2 * (center + left + right + up + down)
                        tl.store(A_ptr + i * N + j, result)

def jacobi_2d_triton(A, B, N, TSTEPS):
    BLOCK_SIZE = 16
    grid_size = triton.cdiv(N - 2, BLOCK_SIZE)
    grid = (grid_size, grid_size)
    
    jacobi_2d_kernel[grid](
        A, B, N, TSTEPS, BLOCK_SIZE
    )