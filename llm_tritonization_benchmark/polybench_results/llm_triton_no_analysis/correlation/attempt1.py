import triton
import triton.language as tl
import torch

@triton.jit
def correlation_kernel(
    data_ptr, mean_ptr, stddev_ptr, corr_ptr,
    eps, float_n, M: tl.constexpr, N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate mean for each column
    for j in range(M):
        mean_sum = 0.0
        for i in range(N):
            data_idx = i * M + j
            mean_sum += tl.load(data_ptr + data_idx)
        mean_val = mean_sum / float_n
        tl.store(mean_ptr + j, mean_val)
    
    # Calculate standard deviation for each column
    for j in range(M):
        mean_val = tl.load(mean_ptr + j)
        stddev_sum = 0.0
        for i in range(N):
            data_idx = i * M + j
            data_val = tl.load(data_ptr + data_idx)
            diff = data_val - mean_val
            stddev_sum += diff * diff
        stddev_val = stddev_sum / float_n
        stddev_val = tl.sqrt(stddev_val)
        stddev_val = tl.where(stddev_val <= eps, 1.0, stddev_val)
        tl.store(stddev_ptr + j, stddev_val)
    
    # Center and reduce the column vectors
    sqrt_float_n = tl.sqrt(float_n)
    for i in range(N):
        for j in range(M):
            data_idx = i * M + j
            mean_val = tl.load(mean_ptr + j)
            stddev_val = tl.load(stddev_ptr + j)
            data_val = tl.load(data_ptr + data_idx)
            
            centered_val = data_val - mean_val
            reduced_val = centered_val / (sqrt_float_n * stddev_val)
            tl.store(data_ptr + data_idx, reduced_val)
    
    # Calculate the correlation matrix
    for i in range(M - 1):
        # Set diagonal element
        diag_idx = i * M + i
        tl.store(corr_ptr + diag_idx, 1.0)
        
        for j in range(i + 1, M):
            corr_sum = 0.0
            for k in range(N):
                data_i_idx = k * M + i
                data_j_idx = k * M + j
                data_i_val = tl.load(data_ptr + data_i_idx)
                data_j_val = tl.load(data_ptr + data_j_idx)
                corr_sum += data_i_val * data_j_val
            
            corr_idx_ij = i * M + j
            corr_idx_ji = j * M + i
            tl.store(corr_ptr + corr_idx_ij, corr_sum)
            tl.store(corr_ptr + corr_idx_ji, corr_sum)
    
    # Set final diagonal element
    final_diag_idx = (M - 1) * M + (M - 1)
    tl.store(corr_ptr + final_diag_idx, 1.0)

def correlation_triton(corr, data, mean, stddev, eps, float_n, M, N):
    BLOCK_SIZE = 32
    
    correlation_kernel[(1,)](
        data, mean, stddev, corr,
        eps, float_n, M, N, BLOCK_SIZE
    )