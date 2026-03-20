import triton
import triton.language as tl
import torch

@triton.jit
def lu_kernel(A_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Single thread block processes entire computation sequentially in i
    for i in range(N):
        # Phase 1: j < i
        j_offsets = tl.arange(0, BLOCK_SIZE)
        
        for j_start in range(0, i, BLOCK_SIZE):
            j_indices = j_start + j_offsets
            j_mask = j_indices < i
            
            # Check if any elements in this block are valid
            valid_elements = tl.sum(j_mask)
            if valid_elements == 0:
                continue
                
            # Initialize accumulator for A[i][j] values
            acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            
            # Inner k loop: accumulate A[i][k] * A[k][j]
            for k in range(j_start):  # k < j, and j starts at j_start
                if k < i:  # Additional safety check
                    a_ik = tl.load(A_ptr + i * N + k)
                    k_row_base = k * N + j_start
                    k_offsets = tl.arange(0, BLOCK_SIZE)
                    k_mask = (j_start + k_offsets < i) & (k_offsets + j_start < N) & (k < j_start + k_offsets)
                    a_kj = tl.load(A_ptr + k_row_base + k_offsets, mask=k_mask, other=0.0)
                    acc -= a_ik * a_kj
            
            # Load current A[i][j] values and add accumulation
            i_row_base = i * N + j_start
            current_aij = tl.load(A_ptr + i_row_base + j_offsets, mask=j_mask, other=0.0)
            current_aij += acc
            
            # Divide by diagonal elements A[j][j]
            for j_idx in range(BLOCK_SIZE):
                if j_start + j_idx < i:
                    diag_val = tl.load(A_ptr + (j_start + j_idx) * N + (j_start + j_idx))
                    if j_start + j_idx < N:
                        current_val = tl.load(A_ptr + i * N + (j_start + j_idx))
                        new_val = current_val / diag_val
                        tl.store(A_ptr + i * N + (j_start + j_idx), new_val)
        
        # Phase 2: j >= i  
        for j_start in range(i, N, BLOCK_SIZE):
            j_indices = j_start + j_offsets
            j_mask = j_indices < N
            
            # Check if any elements in this block are valid
            valid_elements = tl.sum(j_mask)
            if valid_elements == 0:
                continue
                
            # Initialize accumulator
            acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            
            # Inner k loop: k < i
            for k in range(i):
                a_ik = tl.load(A_ptr + i * N + k)
                k_row_base = k * N + j_start
                k_offsets = tl.arange(0, BLOCK_SIZE)
                k_mask = (j_start + k_offsets < N) & j_mask
                a_kj = tl.load(A_ptr + k_row_base + k_offsets, mask=k_mask, other=0.0)
                acc -= a_ik * a_kj
            
            # Load current A[i][j] and update
            i_row_base = i * N + j_start
            current_aij = tl.load(A_ptr + i_row_base + j_offsets, mask=j_mask, other=0.0)
            current_aij += acc
            tl.store(A_ptr + i_row_base + j_offsets, current_aij, mask=j_mask)

def lu_triton(A, N):
    # Use single thread block to handle WAR dependencies
    BLOCK_SIZE = min(triton.next_power_of_2(N), 128)
    
    grid = (1,)
    lu_kernel[grid](A, N, BLOCK_SIZE)