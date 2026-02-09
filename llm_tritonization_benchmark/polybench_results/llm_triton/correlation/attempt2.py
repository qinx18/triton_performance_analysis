import triton
import triton.language as tl
import torch

@triton.jit
def correlation_kernel(
    data_ptr, mean_ptr, stddev_ptr, corr_ptr,
    M, N, float_n, eps,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID for row (i) and column (j)
    pid_i = tl.program_id(axis=0)
    pid_j = tl.program_id(axis=1)
    
    # Calculate global indices
    i_start = pid_i * BLOCK_SIZE
    j_start = pid_j * BLOCK_SIZE
    
    # Create offset vectors
    i_offsets = tl.arange(0, BLOCK_SIZE)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    i_indices = i_start + i_offsets
    j_indices = j_start + j_offsets
    
    # Masks for bounds checking
    i_mask = i_indices < N
    j_mask = j_indices < M
    
    # Step 1: Calculate mean for each column j
    for jj in range(BLOCK_SIZE):
        j_idx = j_start + jj
        if j_idx >= M:
            continue
            
        mean_sum = 0.0
        for ii in range(N):
            data_idx = ii * M + j_idx
            data_val = tl.load(data_ptr + data_idx)
            mean_sum += data_val
        
        mean_val = mean_sum / float_n
        tl.store(mean_ptr + j_idx, mean_val)
    
    # Step 2: Calculate stddev for each column j
    for jj in range(BLOCK_SIZE):
        j_idx = j_start + jj
        if j_idx >= M:
            continue
            
        mean_val = tl.load(mean_ptr + j_idx)
        stddev_sum = 0.0
        
        for ii in range(N):
            data_idx = ii * M + j_idx
            data_val = tl.load(data_ptr + data_idx)
            diff = data_val - mean_val
            stddev_sum += diff * diff
        
        stddev_val = stddev_sum / float_n
        stddev_val = tl.sqrt(stddev_val)
        stddev_val = tl.where(stddev_val <= eps, 1.0, stddev_val)
        tl.store(stddev_ptr + j_idx, stddev_val)
    
    # Step 3: Center and reduce data
    for ii in range(BLOCK_SIZE):
        i_idx = i_start + ii
        if i_idx >= N:
            continue
            
        for jj in range(BLOCK_SIZE):
            j_idx = j_start + jj
            if j_idx >= M:
                continue
                
            data_idx = i_idx * M + j_idx
            data_val = tl.load(data_ptr + data_idx)
            mean_val = tl.load(mean_ptr + j_idx)
            stddev_val = tl.load(stddev_ptr + j_idx)
            
            normalized_val = (data_val - mean_val) / (tl.sqrt(float_n) * stddev_val)
            tl.store(data_ptr + data_idx, normalized_val)
    
    # Step 4: Calculate correlation matrix
    for ii in range(BLOCK_SIZE):
        i_idx = i_start + ii
        if i_idx >= M - 1:
            continue
            
        # Set diagonal
        diag_idx = i_idx * M + i_idx
        tl.store(corr_ptr + diag_idx, 1.0)
        
        for jj in range(BLOCK_SIZE):
            j_idx = j_start + jj
            if j_idx <= i_idx or j_idx >= M:
                continue
                
            corr_sum = 0.0
            for kk in range(N):
                data_i_idx = kk * M + i_idx
                data_j_idx = kk * M + j_idx
                data_i_val = tl.load(data_ptr + data_i_idx)
                data_j_val = tl.load(data_ptr + data_j_idx)
                corr_sum += data_i_val * data_j_val
            
            # Store symmetric values
            corr_ij_idx = i_idx * M + j_idx
            corr_ji_idx = j_idx * M + i_idx
            tl.store(corr_ptr + corr_ij_idx, corr_sum)
            tl.store(corr_ptr + corr_ji_idx, corr_sum)

@triton.jit
def set_last_diagonal_kernel(corr_ptr, M):
    # Set the last diagonal element
    last_diag_idx = (M - 1) * M + (M - 1)
    tl.store(corr_ptr + last_diag_idx, 1.0)

def correlation_triton(corr, data, mean, stddev, eps, float_n, M, N):
    BLOCK_SIZE = 16
    
    # Launch main kernel
    grid = (triton.cdiv(N, BLOCK_SIZE), triton.cdiv(M, BLOCK_SIZE))
    correlation_kernel[grid](
        data, mean, stddev, corr,
        M, N, float_n, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Set the last diagonal element
    set_last_diagonal_kernel[(1,)](corr, M)