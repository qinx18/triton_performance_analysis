import triton
import triton.language as tl
import torch

@triton.jit
def nussinov_kernel(seq_ptr, table_ptr, table_copy_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Get the current diagonal position
    pid = tl.program_id(0)
    
    # For each i from N-1 down to 0
    for i in range(N-1, -1, -1):
        # For each j from i+1 to N-1, process in blocks
        for j_start in range(i+1, N, BLOCK_SIZE):
            # Calculate block offsets
            block_offsets = tl.arange(0, BLOCK_SIZE)
            j_offsets = j_start + block_offsets
            j_mask = (j_offsets < N) & (j_offsets > i)
            
            if not tl.sum(j_mask):
                continue
            
            # Process each j in the block
            for j_idx in range(BLOCK_SIZE):
                j = j_start + j_idx
                if j >= N or j <= i:
                    continue
                
                # Load current table value
                table_idx = i * N + j
                current_val = tl.load(table_ptr + table_idx)
                
                # if (j-1>=0) table[i][j] = max_score(table[i][j], table[i][j-1]);
                if j - 1 >= 0:
                    left_idx = i * N + (j - 1)
                    left_val = tl.load(table_copy_ptr + left_idx)
                    current_val = tl.maximum(current_val, left_val)
                
                # if (i+1<N) table[i][j] = max_score(table[i][j], table[i+1][j]);
                if i + 1 < N:
                    down_idx = (i + 1) * N + j
                    down_val = tl.load(table_copy_ptr + down_idx)
                    current_val = tl.maximum(current_val, down_val)
                
                # if (j-1>=0 && i+1<N)
                if (j - 1 >= 0) & (i + 1 < N):
                    diag_idx = (i + 1) * N + (j - 1)
                    diag_val = tl.load(table_copy_ptr + diag_idx)
                    
                    # if (i<j-1) table[i][j] = max_score(table[i][j], table[i+1][j-1]+match(seq[i], seq[j]));
                    if i < j - 1:
                        seq_i = tl.load(seq_ptr + i)
                        seq_j = tl.load(seq_ptr + j)
                        match_val = tl.where((seq_i + seq_j) == 3, 1, 0)
                        current_val = tl.maximum(current_val, diag_val + match_val)
                    else:
                        current_val = tl.maximum(current_val, diag_val)
                
                # for (k=i+1; k<j; k++) table[i][j] = max_score(table[i][j], table[i][k] + table[k+1][j]);
                for k in range(i + 1, j):
                    left_k_idx = i * N + k
                    right_k_idx = (k + 1) * N + j
                    left_k_val = tl.load(table_copy_ptr + left_k_idx)
                    right_k_val = tl.load(table_copy_ptr + right_k_idx)
                    current_val = tl.maximum(current_val, left_k_val + right_k_val)
                
                # Store the result
                tl.store(table_ptr + table_idx, current_val)

def nussinov_triton(seq, table, N):
    # Create read-only copy of table to handle WAR dependencies
    table_copy = table.clone()
    
    # Launch kernel with a single thread block
    BLOCK_SIZE = 32
    grid = (1,)
    
    nussinov_kernel[grid](
        seq, table, table_copy, N, BLOCK_SIZE
    )
    
    return table