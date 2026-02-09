import triton
import triton.language as tl
import torch

@triton.jit
def symm_kernel(A_ptr, B_ptr, C_ptr, alpha, beta, M, N,
                A_stride0, A_stride1, B_stride0, B_stride1, C_stride0, C_stride1,
                BLOCK_SIZE_J: tl.constexpr):
    
    # Get program ID for j dimension
    pid_j = tl.program_id(0)
    
    # Compute j offsets for this block
    j_offsets = pid_j * BLOCK_SIZE_J + tl.arange(0, BLOCK_SIZE_J)
    j_mask = j_offsets < N
    
    # Sequential loop over i
    for i in range(M):
        # Load B[i][j] values for this block
        B_ptrs = B_ptr + i * B_stride0 + j_offsets * B_stride1
        B_vals = tl.load(B_ptrs, mask=j_mask, other=0.0)
        
        # Load A[i][i] (scalar)
        A_ii_ptr = A_ptr + i * A_stride0 + i * A_stride1
        A_ii = tl.load(A_ii_ptr)
        
        # Initialize temp2 accumulator
        temp2 = tl.zeros([BLOCK_SIZE_J], dtype=tl.float32)
        
        # Inner loop over k < i
        for k in range(i):
            # Load A[i][k] (scalar)
            A_ik_ptr = A_ptr + i * A_stride0 + k * A_stride1
            A_ik = tl.load(A_ik_ptr)
            
            # Load B[k][j] values
            B_k_ptrs = B_ptr + k * B_stride0 + j_offsets * B_stride1
            B_k_vals = tl.load(B_k_ptrs, mask=j_mask, other=0.0)
            
            # Update C[k][j] += alpha * B[i][j] * A[i][k]
            C_k_ptrs = C_ptr + k * C_stride0 + j_offsets * C_stride1
            C_k_vals = tl.load(C_k_ptrs, mask=j_mask, other=0.0)
            C_k_vals = C_k_vals + alpha * B_vals * A_ik
            tl.store(C_k_ptrs, C_k_vals, mask=j_mask)
            
            # Accumulate temp2 += B[k][j] * A[i][k]
            temp2 = temp2 + B_k_vals * A_ik
        
        # Update C[i][j] = beta * C[i][j] + alpha * B[i][j] * A[i][i] + alpha * temp2
        C_i_ptrs = C_ptr + i * C_stride0 + j_offsets * C_stride1
        C_i_vals = tl.load(C_i_ptrs, mask=j_mask, other=0.0)
        C_i_vals = beta * C_i_vals + alpha * B_vals * A_ii + alpha * temp2
        tl.store(C_i_ptrs, C_i_vals, mask=j_mask)

def symm_triton(A, B, C, alpha, beta, M, N):
    BLOCK_SIZE_J = 32
    
    # Calculate grid size for j dimension
    grid_j = triton.cdiv(N, BLOCK_SIZE_J)
    
    # Launch kernel
    symm_kernel[(grid_j,)](
        A, B, C, alpha, beta, M, N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_J=BLOCK_SIZE_J
    )