import triton
import triton.language as tl
import torch

@triton.jit
def covariance_kernel(data_ptr, mean_ptr, cov_ptr, float_n, M: tl.constexpr, N: tl.constexpr):
    # Step 1: Compute means
    for j in range(M):
        mean_val = 0.0
        for i in range(N):
            mean_val += tl.load(data_ptr + i * M + j)
        mean_val /= float_n
        tl.store(mean_ptr + j, mean_val)
    
    # Step 2: Subtract means from data
    for i in range(N):
        for j in range(M):
            data_val = tl.load(data_ptr + i * M + j)
            mean_val = tl.load(mean_ptr + j)
            tl.store(data_ptr + i * M + j, data_val - mean_val)
    
    # Step 3: Compute covariance matrix
    for i in range(M):
        for j in range(i, M):
            cov_val = 0.0
            for k in range(N):
                data_ki = tl.load(data_ptr + k * M + i)
                data_kj = tl.load(data_ptr + k * M + j)
                cov_val += data_ki * data_kj
            cov_val /= (float_n - 1.0)
            tl.store(cov_ptr + i * M + j, cov_val)
            tl.store(cov_ptr + j * M + i, cov_val)

def covariance_triton(cov, data, mean, float_n, M, N):
    grid = (1,)
    covariance_kernel[grid](
        data, mean, cov, float_n, M, N
    )