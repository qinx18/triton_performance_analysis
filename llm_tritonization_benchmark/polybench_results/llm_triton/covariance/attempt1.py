import triton
import triton.language as tl
import torch

@triton.jit
def covariance_kernel(
    cov_ptr, data_ptr, mean_ptr,
    float_n, M, N,
    BLOCK_SIZE: tl.constexpr
):
    # Step 1: Compute mean for each column j
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for j_block_start in range(0, M, BLOCK_SIZE):
        j_current = j_block_start + j_offsets
        j_mask = j_current < M
        
        # Initialize mean values to 0
        mean_vals = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        
        # Sum over all rows for each column
        for i in range(N):
            data_indices = i * M + j_current
            data_vals = tl.load(data_ptr + data_indices, mask=j_mask, other=0.0)
            mean_vals += data_vals
        
        # Divide by N to get mean
        mean_vals = mean_vals / float_n
        
        # Store mean values
        tl.store(mean_ptr + j_current, mean_vals, mask=j_mask)
    
    # Step 2: Subtract mean from data
    for i in range(N):
        for j_block_start in range(0, M, BLOCK_SIZE):
            j_current = j_block_start + j_offsets
            j_mask = j_current < M
            
            data_indices = i * M + j_current
            data_vals = tl.load(data_ptr + data_indices, mask=j_mask, other=0.0)
            mean_vals = tl.load(mean_ptr + j_current, mask=j_mask, other=0.0)
            
            centered_vals = data_vals - mean_vals
            tl.store(data_ptr + data_indices, centered_vals, mask=j_mask)
    
    # Step 3: Compute covariance matrix (upper triangular)
    for i in range(M):
        for j_block_start in range(i, M, BLOCK_SIZE):
            j_current = j_block_start + j_offsets
            j_mask = (j_current < M) & (j_current >= i)
            
            # Initialize covariance values
            cov_vals = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            
            # Sum products over all samples
            for k in range(N):
                data_i_vals = tl.full([BLOCK_SIZE], 0.0, dtype=tl.float32)
                data_i_val = tl.load(data_ptr + k * M + i)
                data_i_vals = data_i_val
                
                data_j_vals = tl.load(data_ptr + k * M + j_current, mask=j_mask, other=0.0)
                
                cov_vals += data_i_vals * data_j_vals
            
            # Divide by (N-1)
            cov_vals = cov_vals / (float_n - 1.0)
            
            # Store upper triangular values
            cov_indices = i * M + j_current
            tl.store(cov_ptr + cov_indices, cov_vals, mask=j_mask)
            
            # Store symmetric values (lower triangular)
            valid_j_mask = j_mask & (j_current != i)  # Don't duplicate diagonal
            if tl.sum(valid_j_mask.to(tl.int32)) > 0:
                sym_indices = j_current * M + i
                tl.store(cov_ptr + sym_indices, cov_vals, mask=valid_j_mask)

def covariance_triton(cov, data, mean, float_n, M, N):
    BLOCK_SIZE = 32
    
    # Launch kernel with single thread block since we handle blocking internally
    grid = (1,)
    
    covariance_kernel[grid](
        cov, data, mean,
        float_n, M, N,
        BLOCK_SIZE=BLOCK_SIZE
    )