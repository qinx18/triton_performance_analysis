import triton
import triton.language as tl
import torch

@triton.jit
def gramschmidt_kernel(A_ptr, Q_ptr, R_ptr, M, N, stride_A_0, stride_A_1, stride_Q_0, stride_Q_1, stride_R_0, stride_R_1, BLOCK_SIZE_I: tl.constexpr):
    i_offsets = tl.arange(0, BLOCK_SIZE_I)
    
    for k in range(N):
        # Compute norm squared
        nrm_sum = 0.0
        for i_block in range(0, M, BLOCK_SIZE_I):
            i_indices = i_block + i_offsets
            i_mask = i_indices < M
            
            a_ptrs = A_ptr + i_indices * stride_A_0 + k * stride_A_1
            a_vals = tl.load(a_ptrs, mask=i_mask, other=0.0)
            nrm_sum += tl.sum(a_vals * a_vals)
        
        # Compute R[k,k]
        r_kk = tl.sqrt(nrm_sum)
        r_kk_ptr = R_ptr + k * stride_R_0 + k * stride_R_1
        tl.store(r_kk_ptr, r_kk)
        
        # Compute Q[:,k] = A[:,k] / R[k,k]
        for i_block in range(0, M, BLOCK_SIZE_I):
            i_indices = i_block + i_offsets
            i_mask = i_indices < M
            
            a_ptrs = A_ptr + i_indices * stride_A_0 + k * stride_A_1
            q_ptrs = Q_ptr + i_indices * stride_Q_0 + k * stride_Q_1
            
            a_vals = tl.load(a_ptrs, mask=i_mask, other=0.0)
            q_vals = a_vals / r_kk
            tl.store(q_ptrs, q_vals, mask=i_mask)
        
        # Process remaining columns
        for j in range(k + 1, N):
            # Initialize R[k,j] to 0
            r_kj_ptr = R_ptr + k * stride_R_0 + j * stride_R_1
            tl.store(r_kj_ptr, 0.0)
            
            # Compute R[k,j] = Q[:,k]^T * A[:,j]
            r_kj_sum = 0.0
            for i_block in range(0, M, BLOCK_SIZE_I):
                i_indices = i_block + i_offsets
                i_mask = i_indices < M
                
                q_ptrs = Q_ptr + i_indices * stride_Q_0 + k * stride_Q_1
                a_ptrs = A_ptr + i_indices * stride_A_0 + j * stride_A_1
                
                q_vals = tl.load(q_ptrs, mask=i_mask, other=0.0)
                a_vals = tl.load(a_ptrs, mask=i_mask, other=0.0)
                r_kj_sum += tl.sum(q_vals * a_vals)
            
            # Store R[k,j]
            tl.store(r_kj_ptr, r_kj_sum)
            
            # Update A[:,j] = A[:,j] - Q[:,k] * R[k,j]
            for i_block in range(0, M, BLOCK_SIZE_I):
                i_indices = i_block + i_offsets
                i_mask = i_indices < M
                
                a_ptrs = A_ptr + i_indices * stride_A_0 + j * stride_A_1
                q_ptrs = Q_ptr + i_indices * stride_Q_0 + k * stride_Q_1
                
                a_vals = tl.load(a_ptrs, mask=i_mask, other=0.0)
                q_vals = tl.load(q_ptrs, mask=i_mask, other=0.0)
                
                new_a_vals = a_vals - q_vals * r_kj_sum
                tl.store(a_ptrs, new_a_vals, mask=i_mask)

def gramschmidt_triton(A, Q, R, M, N):
    BLOCK_SIZE_I = 64
    
    grid = (1,)
    
    gramschmidt_kernel[grid](
        A, Q, R, M, N,
        A.stride(0), A.stride(1),
        Q.stride(0), Q.stride(1), 
        R.stride(0), R.stride(1),
        BLOCK_SIZE_I=BLOCK_SIZE_I
    )