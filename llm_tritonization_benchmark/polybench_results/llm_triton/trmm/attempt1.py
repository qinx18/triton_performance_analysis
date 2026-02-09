import triton
import triton.language as tl
import torch

@triton.jit
def trmm_kernel(A_ptr, B_ptr, B_copy_ptr, alpha, M, N, 
                BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_J: tl.constexpr):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    i_offset = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    j_offset = pid_j * BLOCK_SIZE_J + tl.arange(0, BLOCK_SIZE_J)
    
    i_mask = i_offset < M
    j_mask = j_offset < N
    
    # Load B values for this block
    B_block = tl.zeros((BLOCK_SIZE_I, BLOCK_SIZE_J), dtype=tl.float32)
    for ii in range(BLOCK_SIZE_I):
        for jj in range(BLOCK_SIZE_J):
            i_idx = pid_i * BLOCK_SIZE_I + ii
            j_idx = pid_j * BLOCK_SIZE_J + jj
            if i_idx < M and j_idx < N:
                b_idx = i_idx * N + j_idx
                B_block = tl.where((ii == tl.arange(0, BLOCK_SIZE_I)[:, None]) & 
                                 (jj == tl.arange(0, BLOCK_SIZE_J)[None, :]),
                                 tl.load(B_copy_ptr + b_idx),
                                 B_block)
    
    # Compute the triangular matrix multiplication
    for ii in range(BLOCK_SIZE_I):
        for jj in range(BLOCK_SIZE_J):
            i_idx = pid_i * BLOCK_SIZE_I + ii
            j_idx = pid_j * BLOCK_SIZE_J + jj
            
            if i_idx < M and j_idx < N:
                accumulator = tl.zeros((), dtype=tl.float32)
                
                # Inner loop: k from i+1 to M-1
                for k in range(i_idx + 1, M):
                    a_idx = k * M + i_idx
                    b_idx = k * N + j_idx
                    a_val = tl.load(A_ptr + a_idx)
                    b_val = tl.load(B_copy_ptr + b_idx)
                    accumulator += a_val * b_val
                
                # Add to original B value and multiply by alpha
                orig_b_idx = i_idx * N + j_idx
                orig_b_val = tl.load(B_copy_ptr + orig_b_idx)
                final_val = alpha * (orig_b_val + accumulator)
                
                tl.store(B_ptr + orig_b_idx, final_val)

def trmm_triton(A, B, alpha, M, N):
    # Create read-only copy of B to handle WAR dependencies
    B_copy = B.clone()
    
    BLOCK_SIZE_I = 8
    BLOCK_SIZE_J = 8
    
    grid_i = triton.cdiv(M, BLOCK_SIZE_I)
    grid_j = triton.cdiv(N, BLOCK_SIZE_J)
    
    grid = (grid_i, grid_j)
    
    trmm_kernel[grid](
        A, B, B_copy, alpha, M, N,
        BLOCK_SIZE_I=BLOCK_SIZE_I,
        BLOCK_SIZE_J=BLOCK_SIZE_J
    )