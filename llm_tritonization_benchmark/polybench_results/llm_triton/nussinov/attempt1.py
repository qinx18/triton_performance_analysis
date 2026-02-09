import triton
import triton.language as tl
import torch

@triton.jit
def nussinov_kernel(seq_ptr, table_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Get the thread ID for the diagonal
    diag_id = tl.program_id(0)
    
    # Process diagonals from bottom-right to top-left
    for diag in range(N-1):
        # Skip if this thread doesn't handle this diagonal
        if diag_id != 0:
            continue
            
        # Process each diagonal
        for d in range(diag + 1):
            i = N - 1 - d
            j = i + diag + 1
            
            if i < 0 or j >= N:
                continue
                
            # Compute table index
            table_idx = i * N + j
            
            # Initialize with current value
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
                
                # if (i<j-1) ... else ...
                if i < j - 1:
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
            tl.store(table_ptr + table_idx, current_val)

def nussinov_triton(seq, table, N):
    # Process the algorithm sequentially since it has dependencies
    for i in range(N-1, -1, -1):
        for j in range(i+1, N):
            # Get current value
            current_val = table[i, j].clone()
            
            # if (j-1>=0) table[i][j] = max_score(table[i][j], table[i][j-1]);
            if j - 1 >= 0:
                current_val = torch.maximum(current_val, table[i, j-1])
            
            # if (i+1<N) table[i][j] = max_score(table[i][j], table[i+1][j]);
            if i + 1 < N:
                current_val = torch.maximum(current_val, table[i+1, j])
            
            # if (j-1>=0 && i+1<N)
            if (j - 1 >= 0) and (i + 1 < N):
                if i < j - 1:
                    # Don't allow adjacent elements to bond
                    match_val = 1 if (seq[i] + seq[j]) == 3 else 0
                    current_val = torch.maximum(current_val, table[i+1, j-1] + match_val)
                else:
                    current_val = torch.maximum(current_val, table[i+1, j-1])
            
            # for (k=i+1; k<j; k++)
            for k in range(i+1, j):
                split_sum = table[i, k] + table[k+1, j]
                current_val = torch.maximum(current_val, split_sum)
            
            # Store the result
            table[i, j] = current_val