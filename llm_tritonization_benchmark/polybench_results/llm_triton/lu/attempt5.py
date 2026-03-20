import triton
import triton.language as tl
import torch

@triton.jit
def lu_kernel(A_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    for i in range(N):
        # Phase 1: j < i (compute L matrix elements)
        j_offsets = tl.arange(0, BLOCK_SIZE)
        
        for j_start in range(0, i, BLOCK_SIZE):
            j_indices = j_start + j_offsets
            j_mask = j_indices < i
            
            if j_start >= i:
                break
                
            # Compute sum for each j: sum over k < j
            for j_idx in range(j_start, min(j_start + BLOCK_SIZE, i)):
                acc = 0.0
                for k in range(j_idx):
                    a_ik = tl.load(A_ptr + i * N + k)
                    a_kj = tl.load(A_ptr + k * N + j_idx)
                    acc -= a_ik * a_kj
                
                # Update A[i][j]
                current_val = tl.load(A_ptr + i * N + j_idx)
                current_val += acc
                
                # Divide by A[j][j]
                diag_val = tl.load(A_ptr + j_idx * N + j_idx)
                final_val = current_val / diag_val
                tl.store(A_ptr + i * N + j_idx, final_val)
        
        # Phase 2: j >= i (compute U matrix elements)
        for j_start in range(i, N, BLOCK_SIZE):
            j_indices = j_start + j_offsets
            j_mask = j_indices < N
            
            if j_start >= N:
                break
                
            for j_idx in range(j_start, min(j_start + BLOCK_SIZE, N)):
                acc = 0.0
                for k in range(i):
                    a_ik = tl.load(A_ptr + i * N + k)
                    a_kj = tl.load(A_ptr + k * N + j_idx)
                    acc -= a_ik * a_kj
                
                # Update A[i][j]
                current_val = tl.load(A_ptr + i * N + j_idx)
                final_val = current_val + acc
                tl.store(A_ptr + i * N + j_idx, final_val)

def lu_triton(A, N):
    BLOCK_SIZE = min(triton.next_power_of_2(N), 128)
    
    grid = (1,)
    lu_kernel[grid](A, N, BLOCK_SIZE)