import triton
import triton.language as tl
import torch

@triton.jit
def nussinov_kernel(seq_ptr, table_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Sequential execution with grid=(1,) as specified
    for i_block in range(N):
        i = N - 1 - i_block  # Reverse iteration: i from N-1 down to 0
        
        # Process all j values for this i
        num_j = N - (i + 1)
        if num_j <= 0:
            continue
            
        for j_start in range(i + 1, N, BLOCK_SIZE):
            j_end = min(j_start + BLOCK_SIZE, N)
            
            for j_offset in range(BLOCK_SIZE):
                j = j_start + j_offset
                if j >= N:
                    continue
                if j <= i:
                    continue
                
                # Initialize current value
                current_val = tl.load(table_ptr + i * N + j)
                
                # if (j-1>=0)
                #    table[i][j] = max_score(table[i][j], table[i][j-1]);
                if j - 1 >= 0:
                    val1 = tl.load(table_ptr + i * N + (j - 1))
                    current_val = tl.maximum(current_val, val1)
                
                # if (i+1<N)
                #    table[i][j] = max_score(table[i][j], table[i+1][j]);
                if i + 1 < N:
                    val2 = tl.load(table_ptr + (i + 1) * N + j)
                    current_val = tl.maximum(current_val, val2)
                
                # if (j-1>=0 && i+1<N) {
                #   if (i<j-1)
                #      table[i][j] = max_score(table[i][j], table[i+1][j-1]+match(seq[i], seq[j]));
                #   else
                #      table[i][j] = max_score(table[i][j], table[i+1][j-1]);
                # }
                if (j - 1 >= 0) & (i + 1 < N):
                    val3 = tl.load(table_ptr + (i + 1) * N + (j - 1))
                    if i < j - 1:
                        seq_i = tl.load(seq_ptr + i)
                        seq_j = tl.load(seq_ptr + j)
                        match_val = tl.where((seq_i + seq_j) == 3, 1, 0)
                        val3 = val3 + match_val
                    current_val = tl.maximum(current_val, val3)
                
                # for (k=i+1; k<j; k++) {
                #    table[i][j] = max_score(table[i][j], table[i][k] + table[k+1][j]);
                # }
                for k in range(i + 1, j):
                    val_ik = tl.load(table_ptr + i * N + k)
                    val_k1j = tl.load(table_ptr + (k + 1) * N + j)
                    current_val = tl.maximum(current_val, val_ik + val_k1j)
                
                # Store the result
                tl.store(table_ptr + i * N + j, current_val)

def nussinov_triton(seq, table, N):
    BLOCK_SIZE = 32
    
    # Single CTA execution as specified for this kernel
    grid = (1,)
    
    nussinov_kernel[grid](
        seq,
        table, 
        N,
        BLOCK_SIZE
    )