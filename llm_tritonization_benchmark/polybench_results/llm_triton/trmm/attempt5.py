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
    
    # Process k values
    k_offsets = tl.arange(0, BLOCK_I)
    
    for k_block in range(0, triton.cdiv(M, BLOCK_I)):
        k_base = k_block * BLOCK_I
        k_indices = k_base + k_offsets
        k_valid = k_indices < M
        
        # For each i in current block
        for i_local in range(BLOCK_I):
            i_global = pid_i * BLOCK_I + i_local
            if i_global >= M:
                continue
                
            # Check if k > i
            k_mask = k_valid & (k_indices > i_global)
            
            if tl.sum(k_mask.to(tl.int32)) == 0:
                continue
            
            # Load A[k, i] values
            a_ptrs = A_ptr + k_indices * stride_A_0 + i_global * stride_A_1
            a_vals = tl.load(a_ptrs, mask=k_mask, other=0.0)
            
            # Load B[k, j] values
            b_k_ptrs = B_ptr + k_indices[:, None] * stride_B_0 + j_offsets[None, :] * stride_B_1
            b_k_mask = k_mask[:, None] & j_mask[None, :]
            b_k_vals = tl.load(b_k_ptrs, mask=b_k_mask, other=0.0)
            
            # Compute products and sum over k
            products = a_vals[:, None] * b_k_vals
            k_sum = tl.sum(products, axis=0)
            
            # Update result for this i row
            if i_local < BLOCK_I:
                result = tl.where((tl.arange(0, BLOCK_I) == i_local)[:, None], 
                                result + k_sum[None, :], result)
    
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