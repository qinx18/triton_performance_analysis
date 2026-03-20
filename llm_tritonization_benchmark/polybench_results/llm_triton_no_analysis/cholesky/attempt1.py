import triton
import triton.language as tl
import torch

@triton.jit
def cholesky_kernel(A_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Each program handles one row
    i = tl.program_id(0)
    
    if i >= N:
        return
    
    # Process j < i (off-diagonal elements)
    for j in range(i):
        # Accumulate A[i][j] -= A[i][k] * A[j][k] for k < j
        accumulator = 0.0
        
        # Vectorized computation for k < j
        k_offsets = tl.arange(0, BLOCK_SIZE)
        for k_start in range(0, j, BLOCK_SIZE):
            k_indices = k_start + k_offsets
            k_mask = (k_indices < j)
            
            # Load A[i][k] and A[j][k]
            a_i_k_offsets = i * N + k_indices
            a_j_k_offsets = j * N + k_indices
            
            a_i_k = tl.load(A_ptr + a_i_k_offsets, mask=k_mask, other=0.0)
            a_j_k = tl.load(A_ptr + a_j_k_offsets, mask=k_mask, other=0.0)
            
            # Accumulate products
            products = a_i_k * a_j_k
            accumulator += tl.sum(products)
        
        # Update A[i][j]
        a_ij_offset = i * N + j
        a_ij = tl.load(A_ptr + a_ij_offset)
        a_ij -= accumulator
        
        # Load A[j][j] and divide
        a_jj_offset = j * N + j
        a_jj = tl.load(A_ptr + a_jj_offset)
        a_ij /= a_jj
        
        # Store updated A[i][j]
        tl.store(A_ptr + a_ij_offset, a_ij)
    
    # Process diagonal element A[i][i]
    diagonal_accumulator = 0.0
    
    # Vectorized computation for k < i
    k_offsets = tl.arange(0, BLOCK_SIZE)
    for k_start in range(0, i, BLOCK_SIZE):
        k_indices = k_start + k_offsets
        k_mask = (k_indices < i)
        
        # Load A[i][k]
        a_i_k_offsets = i * N + k_indices
        a_i_k = tl.load(A_ptr + a_i_k_offsets, mask=k_mask, other=0.0)
        
        # Accumulate squares
        squares = a_i_k * a_i_k
        diagonal_accumulator += tl.sum(squares)
    
    # Update A[i][i]
    a_ii_offset = i * N + i
    a_ii = tl.load(A_ptr + a_ii_offset)
    a_ii -= diagonal_accumulator
    a_ii = tl.sqrt(a_ii)
    
    # Store updated A[i][i]
    tl.store(A_ptr + a_ii_offset, a_ii)

def cholesky_triton(A, N):
    BLOCK_SIZE = 128
    
    # Launch kernel with one program per row
    grid = (N,)
    
    cholesky_kernel[grid](
        A, N, BLOCK_SIZE
    )