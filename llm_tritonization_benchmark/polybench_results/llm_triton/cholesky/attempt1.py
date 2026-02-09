import triton
import triton.language as tl
import torch

@triton.jit
def cholesky_kernel(A_ptr, N, stride_0, stride_1):
    for i in range(N):
        # j < i case
        for j in range(i):
            # Inner k loop: A[i][j] -= A[i][k] * A[j][k]
            k_block_size = min(32, j) if j > 0 else 0
            if k_block_size > 0:
                k_offsets = tl.arange(0, 32)
                sum_val = 0.0
                
                for k_start in range(0, j, 32):
                    k_end = min(k_start + 32, j)
                    current_k_offsets = k_start + k_offsets
                    mask = current_k_offsets < j
                    
                    # Load A[i][k]
                    a_i_k_ptr = A_ptr + i * stride_0 + current_k_offsets * stride_1
                    a_i_k = tl.load(a_i_k_ptr, mask=mask, other=0.0)
                    
                    # Load A[j][k]
                    a_j_k_ptr = A_ptr + j * stride_0 + current_k_offsets * stride_1
                    a_j_k = tl.load(a_j_k_ptr, mask=mask, other=0.0)
                    
                    # Accumulate products
                    products = a_i_k * a_j_k
                    sum_val += tl.sum(tl.where(mask, products, 0.0))
                
                # Update A[i][j]
                a_i_j_ptr = A_ptr + i * stride_0 + j * stride_1
                current_val = tl.load(a_i_j_ptr)
                current_val -= sum_val
                tl.store(a_i_j_ptr, current_val)
            
            # Divide by A[j][j]
            a_j_j_ptr = A_ptr + j * stride_0 + j * stride_1
            a_j_j = tl.load(a_j_j_ptr)
            
            a_i_j_ptr = A_ptr + i * stride_0 + j * stride_1
            a_i_j = tl.load(a_i_j_ptr)
            a_i_j /= a_j_j
            tl.store(a_i_j_ptr, a_i_j)
        
        # i == j case
        sum_val = 0.0
        if i > 0:
            k_offsets = tl.arange(0, 32)
            
            for k_start in range(0, i, 32):
                current_k_offsets = k_start + k_offsets
                mask = current_k_offsets < i
                
                # Load A[i][k]
                a_i_k_ptr = A_ptr + i * stride_0 + current_k_offsets * stride_1
                a_i_k = tl.load(a_i_k_ptr, mask=mask, other=0.0)
                
                # Accumulate squares
                squares = a_i_k * a_i_k
                sum_val += tl.sum(tl.where(mask, squares, 0.0))
        
        # Update A[i][i]
        a_i_i_ptr = A_ptr + i * stride_0 + i * stride_1
        a_i_i = tl.load(a_i_i_ptr)
        a_i_i -= sum_val
        a_i_i = tl.sqrt(a_i_i)
        tl.store(a_i_i_ptr, a_i_i)

def cholesky_triton(A, N):
    grid = (1,)
    cholesky_kernel[grid](
        A, N,
        A.stride(0), A.stride(1)
    )