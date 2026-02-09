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
    
    # Initialize result matrix
    result = tl.zeros((BLOCK_I, BLOCK_J), dtype=tl.float32)
    
    # Load initial B values
    b_ptrs = B_ptr + i_offsets[:, None] * stride_B_0 + j_offsets[None, :] * stride_B_1
    mask = i_mask[:, None] & j_mask[None, :]
    b_vals = tl.load(b_ptrs, mask=mask, other=0.0)
    result = b_vals
    
    # For each i in the block
    for i_idx in range(BLOCK_I):
        i_global = pid_i * BLOCK_I + i_idx
        if i_global >= M:
            continue
            
        # Compute k range: k from i+1 to M-1
        k_start = i_global + 1
        if k_start >= M:
            continue
            
        # Process k values
        for k_base in range(k_start, M, 32):  # Process k in chunks
            k_offsets = k_base + tl.arange(0, 32)
            k_mask = (k_offsets < M) & (k_offsets >= k_start)
            
            if not tl.any(k_mask):
                continue
                
            # Load A[k, i_global] values
            a_ptrs = A_ptr + k_offsets * stride_A_0 + i_global * stride_A_1
            a_vals = tl.load(a_ptrs, mask=k_mask, other=0.0)
            
            # Load B[k, :] values for current j block
            b_k_ptrs = B_ptr + k_offsets[:, None] * stride_B_0 + j_offsets[None, :] * stride_B_1
            b_k_mask = k_mask[:, None] & j_mask[None, :]
            b_k_vals = tl.load(b_k_ptrs, mask=b_k_mask, other=0.0)
            
            # Accumulate A[k, i] * B[k, j] for this i
            products = a_vals[:, None] * b_k_vals
            # Sum over k dimension, but only for the specific i_idx
            k_contribution = tl.sum(products, axis=0)
            result = tl.where(tl.arange(0, BLOCK_I)[:, None] == i_idx, 
                             result + k_contribution[None, :], 
                             result)
    
    # Apply alpha scaling
    result = alpha * result
    
    # Store results back to B
    b_ptrs = B_ptr + i_offsets[:, None] * stride_B_0 + j_offsets[None, :] * stride_B_1
    mask = i_mask[:, None] & j_mask[None, :]
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