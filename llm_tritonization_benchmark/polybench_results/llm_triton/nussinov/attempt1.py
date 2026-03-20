import triton
import triton.language as tl
import torch

@triton.jit
def nussinov_kernel(seq_ptr, table_ptr, N: tl.constexpr):
    # Single CTA processes entire computation sequentially
    for i in range(N):
        actual_i = N - 1 - i  # Reverse iteration: i from N-1 down to 0
        
        for j in range(actual_i + 1, N):
            table_idx = actual_i * N + j
            
            # Initialize with current value
            current_val = tl.load(table_ptr + table_idx)
            
            # if (j-1>=0) table[i][j] = max_score(table[i][j], table[i][j-1]);
            if j - 1 >= 0:
                left_idx = actual_i * N + (j - 1)
                left_val = tl.load(table_ptr + left_idx)
                current_val = tl.maximum(current_val, left_val)
            
            # if (i+1<N) table[i][j] = max_score(table[i][j], table[i+1][j]);
            if actual_i + 1 < N:
                down_idx = (actual_i + 1) * N + j
                down_val = tl.load(table_ptr + down_idx)
                current_val = tl.maximum(current_val, down_val)
            
            # if (j-1>=0 && i+1<N)
            if j - 1 >= 0 and actual_i + 1 < N:
                diag_idx = (actual_i + 1) * N + (j - 1)
                diag_val = tl.load(table_ptr + diag_idx)
                
                if actual_i < j - 1:  # don't allow adjacent elements to bond
                    seq_i = tl.load(seq_ptr + actual_i)
                    seq_j = tl.load(seq_ptr + j)
                    match_val = tl.where(seq_i + seq_j == 3, 1, 0)
                    current_val = tl.maximum(current_val, diag_val + match_val)
                else:
                    current_val = tl.maximum(current_val, diag_val)
            
            # for (k=i+1; k<j; k++)
            for k in range(actual_i + 1, j):
                left_k_idx = actual_i * N + k
                right_k_idx = (k + 1) * N + j
                left_k_val = tl.load(table_ptr + left_k_idx)
                right_k_val = tl.load(table_ptr + right_k_idx)
                current_val = tl.maximum(current_val, left_k_val + right_k_val)
            
            # Store final result
            tl.store(table_ptr + table_idx, current_val)

def nussinov_triton(seq, table, N):
    # Single CTA handles entire computation
    grid = (1,)
    nussinov_kernel[grid](seq, table, N)