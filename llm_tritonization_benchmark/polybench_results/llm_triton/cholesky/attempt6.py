import triton
import triton.language as tl
import torch
import math

@triton.jit
def cholesky_kernel(A_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    for i in range(N):
        # Phase 1: j < i case
        for j in range(i):
            # Initialize A[i][j] accumulator
            acc = tl.load(A_ptr + i * N + j)
            
            # Reduce over k < j
            k_offsets = tl.arange(0, BLOCK_SIZE)
            for k_start in range(0, j, BLOCK_SIZE):
                k_mask = (k_start + k_offsets) < j
                
                # Load A[i][k] values
                i_k_ptrs = A_ptr + i * N + k_start + k_offsets
                i_k_vals = tl.load(i_k_ptrs, mask=k_mask, other=0.0)
                
                # Load A[j][k] values  
                j_k_ptrs = A_ptr + j * N + k_start + k_offsets
                j_k_vals = tl.load(j_k_ptrs, mask=k_mask, other=0.0)
                
                # Accumulate products
                products = i_k_vals * j_k_vals
                acc -= tl.sum(products)
            
            # Divide by A[j][j]
            diagonal_val = tl.load(A_ptr + j * N + j)
            acc /= diagonal_val
            
            # Store result
            tl.store(A_ptr + i * N + j, acc)
        
        # Phase 2: i == j case (diagonal)
        diagonal_acc = tl.load(A_ptr + i * N + i)
        
        # Reduce over k < i
        k_offsets = tl.arange(0, BLOCK_SIZE)
        for k_start in range(0, i, BLOCK_SIZE):
            k_mask = (k_start + k_offsets) < i
            
            # Load A[i][k] values
            i_k_ptrs = A_ptr + i * N + k_start + k_offsets
            i_k_vals = tl.load(i_k_ptrs, mask=k_mask, other=0.0)
            
            # Accumulate squares
            squares = i_k_vals * i_k_vals
            diagonal_acc -= tl.sum(squares)
        
        # Take square root and store
        diagonal_acc = tl.sqrt(diagonal_acc)
        tl.store(A_ptr + i * N + i, diagonal_acc)

def cholesky_triton(A, N):
    BLOCK_SIZE = 64
    
    grid = (1,)
    cholesky_kernel[grid](A, N, BLOCK_SIZE)