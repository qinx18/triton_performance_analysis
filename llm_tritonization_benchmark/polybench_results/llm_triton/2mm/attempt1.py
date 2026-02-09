import triton
import triton.language as tl
import torch

@triton.jit
def k2mm_kernel(A_ptr, B_ptr, C_ptr, D_ptr, tmp_ptr, alpha, beta, NI, NJ, NK, NL,
                A_stride0, A_stride1, B_stride0, B_stride1, 
                C_stride0, C_stride1, D_stride0, D_stride1,
                tmp_stride0, tmp_stride1,
                BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr):
    
    # Get program IDs
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    # Calculate block offsets
    block_i = pid_i * BLOCK_I
    block_j = pid_j * BLOCK_J
    
    # Create offset vectors once
    i_offsets = tl.arange(0, BLOCK_I)
    j_offsets = tl.arange(0, BLOCK_J)
    
    # Current block indices
    i_indices = block_i + i_offsets
    j_indices = block_j + j_offsets
    
    # Masks for boundary conditions
    i_mask = i_indices < NI
    j_mask = j_indices < NJ
    
    # First loop nest: compute tmp[i][j] = alpha * sum_k(A[i][k] * B[k][j])
    tmp_vals = tl.zeros((BLOCK_I, BLOCK_J), dtype=tl.float32)
    
    for k in range(NK):
        # Load A[i][k] values
        a_ptrs = A_ptr + i_indices[:, None] * A_stride0 + k * A_stride1
        a_vals = tl.load(a_ptrs, mask=i_mask[:, None], other=0.0)
        
        # Load B[k][j] values  
        b_ptrs = B_ptr + k * B_stride0 + j_indices[None, :] * B_stride1
        b_vals = tl.load(b_ptrs, mask=j_mask[None, :], other=0.0)
        
        # Accumulate tmp[i][j] += alpha * A[i][k] * B[k][j]
        tmp_vals += alpha * a_vals * b_vals
    
    # Store tmp values
    tmp_ptrs = tmp_ptr + i_indices[:, None] * tmp_stride0 + j_indices[None, :] * tmp_stride1
    full_mask = i_mask[:, None] & j_mask[None, :]
    tl.store(tmp_ptrs, tmp_vals, mask=full_mask)
    
    # Second loop nest: compute D[i][j] = beta*D[i][j] + sum_k(tmp[i][k] * C[k][j])
    # Only proceed if we have valid i indices for this block
    if block_i < NI:
        # Recalculate j block for NL dimension
        for jl_start in range(0, NL, BLOCK_J):
            jl_indices = jl_start + j_offsets
            jl_mask = jl_indices < NL
            
            # Load and scale D[i][jl] by beta
            d_ptrs = D_ptr + i_indices[:, None] * D_stride0 + jl_indices[None, :] * D_stride1
            d_mask = i_mask[:, None] & jl_mask[None, :]
            d_vals = tl.load(d_ptrs, mask=d_mask, other=0.0) * beta
            
            # Accumulate sum_k(tmp[i][k] * C[k][jl])
            for k in range(NJ):
                # Load tmp[i][k]
                tmp_k_ptrs = tmp_ptr + i_indices[:, None] * tmp_stride0 + k * tmp_stride1
                tmp_k_vals = tl.load(tmp_k_ptrs, mask=i_mask[:, None], other=0.0)
                
                # Load C[k][jl]
                c_ptrs = C_ptr + k * C_stride0 + jl_indices[None, :] * C_stride1
                c_vals = tl.load(c_ptrs, mask=jl_mask[None, :], other=0.0)
                
                # Accumulate
                d_vals += tmp_k_vals * c_vals
            
            # Store D[i][jl]
            tl.store(d_ptrs, d_vals, mask=d_mask)

def k2mm_triton(A, B, C, D, tmp, alpha, beta, NI, NJ, NK, NL):
    # Define block sizes
    BLOCK_I = 16
    BLOCK_J = 16
    
    # Calculate grid dimensions
    grid_i = triton.cdiv(NI, BLOCK_I)
    grid_j = triton.cdiv(NJ, BLOCK_J)
    
    # Launch kernel
    k2mm_kernel[(grid_i, grid_j)](
        A, B, C, D, tmp,
        alpha, beta, NI, NJ, NK, NL,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1), 
        C.stride(0), C.stride(1),
        D.stride(0), D.stride(1),
        tmp.stride(0), tmp.stride(1),
        BLOCK_I=BLOCK_I, BLOCK_J=BLOCK_J
    )