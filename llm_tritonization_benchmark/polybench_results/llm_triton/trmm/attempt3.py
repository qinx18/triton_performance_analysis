import triton
import triton.language as tl
import torch

@triton.jit
def trmm_kernel(
    A_ptr, B_ptr, alpha,
    M, N,
    stride_A_0, stride_A_1,
    stride_B_0, stride_B_1,
    BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr
):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    i_offsets = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    j_offsets = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)
    
    i_mask = i_offsets < M
    j_mask = j_offsets < N
    
    # Load initial B values
    b_ptrs = B_ptr + i_offsets[:, None] * stride_B_0 + j_offsets[None, :] * stride_B_1
    mask = i_mask[:, None] & j_mask[None, :]
    result = tl.load(b_ptrs, mask=mask, other=0.0)
    
    # Process each row in the block
    for i_idx in range(BLOCK_I):
        i_global = pid_i * BLOCK_I + i_idx
        if i_global >= M:
            break
            
        # Compute k range: k from i+1 to M-1
        k_start = i_global + 1
        if k_start >= M:
            break
        
        # Sum accumulator for this specific i
        accumulator = tl.zeros((BLOCK_J,), dtype=tl.float32)
        
        # Process k values in chunks
        k_offsets = tl.arange(0, 32)
        for k_base in range(k_start, M, 32):
            k_indices = k_base + k_offsets
            k_mask = (k_indices < M) & (k_indices >= k_start)
            
            # Load A[k, i_global] values
            a_ptrs = A_ptr + k_indices * stride_A_0 + i_global * stride_A_1
            a_vals = tl.load(a_ptrs, mask=k_mask, other=0.0)
            
            # Load B[k, :] values for current j block
            b_k_ptrs = B_ptr + k_indices[:, None] * stride_B_0 + j_offsets[None, :] * stride_B_1
            b_k_mask = k_mask[:, None] & j_mask[None, :]
            b_k_vals = tl.load(b_k_ptrs, mask=b_k_mask, other=0.0)
            
            # Accumulate A[k, i] * B[k, j]
            products = a_vals[:, None] * b_k_vals
            accumulator += tl.sum(products, axis=0)
        
        # Update result for this specific i row
        i_local_mask = tl.arange(0, BLOCK_I) == i_idx
        result = tl.where(i_local_mask[:, None], result + accumulator[None, :], result)
    
    # Apply alpha scaling
    result = alpha * result
    
    # Store results back to B
    tl.store(b_ptrs, result, mask=mask)

def trmm_triton(A, B, alpha, M, N):
    BLOCK_I = 16
    BLOCK_J = 32
    
    grid = (triton.cdiv(M, BLOCK_I), triton.cdiv(N, BLOCK_J))
    
    trmm_kernel[grid](
        A, B, alpha,
        M, N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        BLOCK_I=BLOCK_I, BLOCK_J=BLOCK_J
    )