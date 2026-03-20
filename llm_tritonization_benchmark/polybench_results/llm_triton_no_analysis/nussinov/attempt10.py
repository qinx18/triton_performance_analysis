import triton
import triton.language as tl
import torch

@triton.jit
def nussinov_kernel(seq_ptr, table_ptr, N: tl.constexpr):
    i = tl.program_id(0)
    
    for j in range(i + 1, N):
        current_idx = i * N + j
        current_val = tl.load(table_ptr + current_idx)
        
        if j - 1 >= 0:
            left_idx = i * N + (j - 1)
            left_val = tl.load(table_ptr + left_idx)
            current_val = tl.maximum(current_val, left_val)
        
        if i + 1 < N:
            down_idx = (i + 1) * N + j
            down_val = tl.load(table_ptr + down_idx)
            current_val = tl.maximum(current_val, down_val)
        
        if (j - 1 >= 0) and (i + 1 < N):
            diag_idx = (i + 1) * N + (j - 1)
            diag_val = tl.load(table_ptr + diag_idx)
            
            if i < j - 1:
                seq_i = tl.load(seq_ptr + i)
                seq_j = tl.load(seq_ptr + j)
                match_val = tl.where((seq_i + seq_j) == 3, 1, 0)
                current_val = tl.maximum(current_val, diag_val + match_val)
            else:
                current_val = tl.maximum(current_val, diag_val)
        
        for k in range(i + 1, j):
            left_k_idx = i * N + k
            right_k_idx = (k + 1) * N + j
            left_k_val = tl.load(table_ptr + left_k_idx)
            right_k_val = tl.load(table_ptr + right_k_idx)
            current_val = tl.maximum(current_val, left_k_val + right_k_val)
        
        tl.store(table_ptr + current_idx, current_val)

def nussinov_triton(seq, table, N):
    for row in range(N-1, -1, -1):
        nussinov_kernel[(1,)](
            seq, table,
            N=N
        )