import torch
import triton
import triton.language as tl

@triton.jit
def trmm_kernel(A_ptr, B_ptr, alpha, M: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program IDs for parallelization
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    # Calculate block start positions
    block_start_i = pid_i * BLOCK_SIZE
    block_start_j = pid_j * BLOCK_SIZE
    
    # Define offset ranges
    offsets_i = tl.arange(0, BLOCK_SIZE)
    offsets_j = tl.arange(0, BLOCK_SIZE)
    
    # Calculate actual indices
    i_indices = block_start_i + offsets_i
    j_indices = block_start_j + offsets_j
    
    # Masks for boundary checking
    mask_i = i_indices < M
    mask_j = j_indices < N
    
    # Process each (i,j) pair in the block
    for i_offset in range(BLOCK_SIZE):
        i = block_start_i + i_offset
        if i >= M:
            break
            
        for j_offset in range(BLOCK_SIZE):
            j = block_start_j + j_offset
            if j >= N:
                break
                
            # Initialize accumulator with current B[i][j]
            b_ij_ptr = B_ptr + i * N + j
            b_ij = tl.load(b_ij_ptr)
            
            # Inner k loop: for k = i+1 to M-1
            for k in range(i + 1, M):
                # B[i][j] += A[k][i] * B[k][j]
                a_ki_ptr = A_ptr + k * M + i
                b_kj_ptr = B_ptr + k * N + j
                
                a_ki = tl.load(a_ki_ptr)
                b_kj = tl.load(b_kj_ptr)
                
                b_ij += a_ki * b_kj
            
            # B[i][j] = alpha * B[i][j]
            b_ij = alpha * b_ij
            
            # Store result back
            tl.store(b_ij_ptr, b_ij)

def trmm_triton(A, B, alpha, M, N):
    # Define block size
    BLOCK_SIZE = 16
    
    # Calculate grid dimensions
    grid_i = triton.cdiv(M, BLOCK_SIZE)
    grid_j = triton.cdiv(N, BLOCK_SIZE)
    
    # Launch kernel
    trmm_kernel[(grid_i, grid_j)](
        A, B, alpha, M, N, BLOCK_SIZE
    )
    
    return B