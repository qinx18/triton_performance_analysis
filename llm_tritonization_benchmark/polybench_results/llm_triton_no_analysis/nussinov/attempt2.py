import triton
import triton.language as tl
import torch

@triton.jit
def nussinov_kernel(seq_ptr, table_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get the diagonal index this block is processing
    diag_idx = tl.program_id(0)
    
    # Process diagonals from bottom-right to top-left
    for i_rev in range(N):
        i = N - 1 - i_rev
        diag_length = N - i - 1
        
        if diag_length <= 0:
            continue
            
        # Only process if this is the correct diagonal
        if diag_idx < diag_length:
            j = i + 1 + diag_idx
            
            if j < N:
                # Compute linear index for table[i][j]
                idx = i * N + j
                
                # Initialize with current value
                current_val = tl.load(table_ptr + idx)
                
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
                if j - 1 >= 0 and i + 1 < N:
                    diag_idx_val = (i + 1) * N + (j - 1)
                    diag_val = tl.load(table_ptr + diag_idx_val)
                    
                    if i < j - 1:  # don't allow adjacent elements to bond
                        seq_i = tl.load(seq_ptr + i)
                        seq_j = tl.load(seq_ptr + j)
                        match_val = tl.where((seq_i + seq_j) == 3, 1, 0)
                        current_val = tl.maximum(current_val, diag_val + match_val)
                    else:
                        current_val = tl.maximum(current_val, diag_val)
                
                # for (k=i+1; k<j; k++)
                for k in range(i + 1, j):
                    left_split_idx = i * N + k
                    right_split_idx = (k + 1) * N + j
                    left_split_val = tl.load(table_ptr + left_split_idx)
                    right_split_val = tl.load(table_ptr + right_split_idx)
                    split_sum = left_split_val + right_split_val
                    current_val = tl.maximum(current_val, split_sum)
                
                # Store the result
                tl.store(table_ptr + idx, current_val)

def nussinov_triton(seq, table, N):
    BLOCK_SIZE = 32
    max_diag_length = N - 1
    grid = (max_diag_length,)
    
    for _ in range(N):
        nussinov_kernel[grid](seq, table, N, BLOCK_SIZE)