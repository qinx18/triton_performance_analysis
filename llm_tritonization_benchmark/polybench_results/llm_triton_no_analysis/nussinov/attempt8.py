import triton
import triton.language as tl
import torch

@triton.jit
def nussinov_kernel(seq_ptr, table_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    i = N - 1 - row
    
    if i < 0:
        return
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    for j_block in range(i + 1, N, BLOCK_SIZE):
        j_offsets = j_block + col_offsets
        j_mask = (j_offsets < N) & (j_offsets > i)
        
        mask_sum = tl.sum(j_mask.to(tl.int32))
        if mask_sum == 0:
            continue
            
        for j_idx in range(BLOCK_SIZE):
            j = j_block + j_idx
            if j >= N or j <= i:
                continue
                
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
            
            if (j - 1 >= 0) & (i + 1 < N):
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
    BLOCK_SIZE = 32
    grid = (N,)
    
    nussinov_kernel[grid](
        seq, table,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE
    )