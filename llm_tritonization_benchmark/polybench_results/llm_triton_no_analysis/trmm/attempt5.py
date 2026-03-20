import torch
import triton
import triton.language as tl

@triton.jit
def trmm_kernel(A_ptr, B_ptr, alpha, M: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    block_start_i = pid_i * BLOCK_SIZE
    block_start_j = pid_j * BLOCK_SIZE
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    i_mask = (block_start_i + i_offsets) < M
    j_mask = (block_start_j + j_offsets) < N
    
    # Load B block
    B_block = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    for ii in range(BLOCK_SIZE):
        i_idx = block_start_i + ii
        if i_idx < M:
            b_ptrs = B_ptr + i_idx * N + block_start_j + j_offsets
            b_vals = tl.load(b_ptrs, mask=j_mask, other=0.0)
            
            # Inner k loop
            for k in range(i_idx + 1, M):
                a_ki = tl.load(A_ptr + k * M + i_idx)
                b_kj_ptrs = B_ptr + k * N + block_start_j + j_offsets
                b_kj_vals = tl.load(b_kj_ptrs, mask=j_mask, other=0.0)
                b_vals = b_vals + a_ki * b_kj_vals
            
            # Apply alpha
            b_vals = alpha * b_vals
            
            # Store back
            tl.store(b_ptrs, b_vals, mask=j_mask)

def trmm_triton(A, B, alpha, M, N):
    BLOCK_SIZE = 16
    
    grid_i = triton.cdiv(M, BLOCK_SIZE)
    grid_j = triton.cdiv(N, BLOCK_SIZE)
    
    trmm_kernel[(grid_i, grid_j)](
        A, B, alpha, M, N, BLOCK_SIZE
    )
    
    return B