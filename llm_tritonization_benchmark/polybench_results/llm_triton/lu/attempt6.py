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
            j_mask = (j_indices < i) & (j_indices >= j_start)
            
            # Inner k loop for phase 1
            for k in range(j_start, min(j_start + BLOCK_SIZE, i)):
                if k < i:
                    k_mask = j_indices < k
                    combined_mask = j_mask & k_mask
                    
                    if tl.sum(combined_mask.to(tl.int32)) > 0:
                        A_i_k = tl.load(A_ptr + i * N + k)
                        A_k_j = tl.load(A_ptr + k * N + j_indices, mask=combined_mask, other=0.0)
                        A_i_j = tl.load(A_ptr + i * N + j_indices, mask=combined_mask, other=0.0)
                        A_i_j = A_i_j - A_i_k * A_k_j
                        tl.store(A_ptr + i * N + j_indices, A_i_j, mask=combined_mask)
            
            # Divide by diagonal elements for phase 1
            if tl.sum(j_mask.to(tl.int32)) > 0:
                A_j_j = tl.load(A_ptr + j_indices * N + j_indices, mask=j_mask, other=1.0)
                A_i_j = tl.load(A_ptr + i * N + j_indices, mask=j_mask, other=0.0)
                A_i_j = A_i_j / A_j_j
                tl.store(A_ptr + i * N + j_indices, A_i_j, mask=j_mask)
        
        # Phase 2: j >= i (compute U matrix elements)
        for j_start in range(i, N, BLOCK_SIZE):
            j_indices = j_start + j_offsets
            j_mask = (j_indices < N) & (j_indices >= j_start)
            
            # Inner k loop for phase 2
            for k in range(i):
                if tl.sum(j_mask.to(tl.int32)) > 0:
                    A_i_k = tl.load(A_ptr + i * N + k)
                    A_k_j = tl.load(A_ptr + k * N + j_indices, mask=j_mask, other=0.0)
                    A_i_j = tl.load(A_ptr + i * N + j_indices, mask=j_mask, other=0.0)
                    A_i_j = A_i_j - A_i_k * A_k_j
                    tl.store(A_ptr + i * N + j_indices, A_i_j, mask=j_mask)

def lu_triton(A, N):
    BLOCK_SIZE = min(triton.next_power_of_2(N), 128)
    grid = (1,)
    lu_kernel[grid](A, N, BLOCK_SIZE)