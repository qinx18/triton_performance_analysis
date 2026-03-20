import triton
import triton.language as tl
import torch

@triton.jit
def nussinov_kernel(seq_ptr, table_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Each block processes one row (i value)
    i_block = tl.program_id(0)
    
    if i_block >= N:
        return
    
    i = N - 1 - i_block  # Reverse iteration: i from N-1 down to 0
    
    # Process all j values for this i in chunks
    for j_start in range(i + 1, N, BLOCK_SIZE):
        j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
        j_mask = (j_offsets < N) & (j_offsets > i)
        
        # Process each j in this block
        for j_idx in range(BLOCK_SIZE):
            j = j_start + j_idx
            if j >= N:
                break
            if j <= i:
                break
                
            table_idx = i * N + j
            current_val = tl.load(table_ptr + table_idx)
            
            # if (j-1>=0) table[i][j] = max_score(table[i][j], table[i][j-1]);
            if j - 1 >= 0:
                left_idx = i * N + (j - 1)
                left_val = tl.load(table_ptr + left_idx)
                current_val = tl.maximum(current_val, left_val)
            
            # if (i+1<N) table[i][j] = max_score(table[i][j], table[i+1][j]);
            if i + 1 < N:
                down_idx = (i + 1) * N + j
                down_val = tl.load(table_ptr + down_idx)
                current_val = tl.maximum(current_val, down_val)
            
            # if (j-1>=0 && i+1<N)
            if (j - 1 >= 0) & (i + 1 < N):
                diag_idx = (i + 1) * N + (j - 1)
                diag_val = tl.load(table_ptr + diag_idx)
                
                if i < j - 1:  # don't allow adjacent elements to bond
                    seq_i = tl.load(seq_ptr + i)
                    seq_j = tl.load(seq_ptr + j)
                    match_val = tl.where(seq_i + seq_j == 3, 1, 0)
                    current_val = tl.maximum(current_val, diag_val + match_val)
                else:
                    current_val = tl.maximum(current_val, diag_val)
            
            # for (k=i+1; k<j; k++)
            for k in range(i + 1, j):
                left_k_idx = i * N + k
                right_k_idx = (k + 1) * N + j
                left_k_val = tl.load(table_ptr + left_k_idx)
                right_k_val = tl.load(table_ptr + right_k_idx)
                current_val = tl.maximum(current_val, left_k_val + right_k_val)
            
            # Store final result
            tl.store(table_ptr + table_idx, current_val)

def nussinov_triton(seq, table, N):
    BLOCK_SIZE = 64
    grid = (N,)  # Each block handles one row (i value)
    nussinov_kernel[grid](seq, table, N, BLOCK_SIZE)