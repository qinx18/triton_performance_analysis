import triton
import triton.language as tl
import torch

@triton.jit
def k2mm_kernel(A_ptr, B_ptr, C_ptr, D_ptr, tmp_ptr,
                alpha, beta,
                NI: tl.constexpr, NJ: tl.constexpr, NK: tl.constexpr, NL: tl.constexpr,
                BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_L: tl.constexpr):
    
    # First stage: tmp[i][j] = alpha * A[i][k] * B[k][j]
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    i_start = pid_i * BLOCK_I
    j_start = pid_j * BLOCK_J
    
    i_offsets = i_start + tl.arange(0, BLOCK_I)
    j_offsets = j_start + tl.arange(0, BLOCK_J)
    
    i_mask = i_offsets < NI
    j_mask = j_offsets < NJ
    
    # Initialize tmp accumulator
    tmp_acc = tl.zeros((BLOCK_I, BLOCK_J), dtype=tl.float32)
    
    # Compute tmp[i][j] = sum over k of alpha * A[i][k] * B[k][j]
    for k in range(0, NK, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < NK
        
        # Load A[i][k]
        A_offsets = i_offsets[:, None] * NK + k_offsets[None, :]
        A_vals = tl.load(A_ptr + A_offsets, mask=i_mask[:, None] & k_mask[None, :], other=0.0)
        
        # Load B[k][j]
        B_offsets = k_offsets[:, None] * NJ + j_offsets[None, :]
        B_vals = tl.load(B_ptr + B_offsets, mask=k_mask[:, None] & j_mask[None, :], other=0.0)
        
        # Accumulate tmp[i][j] += alpha * A[i][k] * B[k][j]
        tmp_acc += alpha * tl.dot(A_vals, B_vals)
    
    # Store tmp results
    tmp_offsets = i_offsets[:, None] * NJ + j_offsets[None, :]
    tl.store(tmp_ptr + tmp_offsets, tmp_acc, mask=i_mask[:, None] & j_mask[None, :])

@triton.jit
def k2mm_second_stage_kernel(tmp_ptr, C_ptr, D_ptr, beta,
                            NI: tl.constexpr, NJ: tl.constexpr, NL: tl.constexpr,
                            BLOCK_I: tl.constexpr, BLOCK_L: tl.constexpr, BLOCK_J: tl.constexpr):
    
    # Second stage: D[i][j] = beta * D[i][j] + tmp[i][k] * C[k][j]
    pid_i = tl.program_id(0)
    pid_l = tl.program_id(1)
    
    i_start = pid_i * BLOCK_I
    l_start = pid_l * BLOCK_L
    
    i_offsets = i_start + tl.arange(0, BLOCK_I)
    l_offsets = l_start + tl.arange(0, BLOCK_L)
    
    i_mask = i_offsets < NI
    l_mask = l_offsets < NL
    
    # Load and scale D[i][l] by beta
    D_offsets = i_offsets[:, None] * NL + l_offsets[None, :]
    D_vals = tl.load(D_ptr + D_offsets, mask=i_mask[:, None] & l_mask[None, :], other=0.0)
    D_acc = beta * D_vals
    
    # Compute D[i][l] += sum over k of tmp[i][k] * C[k][l]
    for k in range(0, NJ, BLOCK_J):
        k_offsets = k + tl.arange(0, BLOCK_J)
        k_mask = k_offsets < NJ
        
        # Load tmp[i][k]
        tmp_offsets = i_offsets[:, None] * NJ + k_offsets[None, :]
        tmp_vals = tl.load(tmp_ptr + tmp_offsets, mask=i_mask[:, None] & k_mask[None, :], other=0.0)
        
        # Load C[k][l]
        C_offsets = k_offsets[:, None] * NL + l_offsets[None, :]
        C_vals = tl.load(C_ptr + C_offsets, mask=k_mask[:, None] & l_mask[None, :], other=0.0)
        
        # Accumulate D[i][l] += tmp[i][k] * C[k][l]
        D_acc += tl.dot(tmp_vals, C_vals)
    
    # Store final D results
    tl.store(D_ptr + D_offsets, D_acc, mask=i_mask[:, None] & l_mask[None, :])

def k2mm_triton(A, B, C, D, tmp, alpha, beta, NI, NJ, NK, NL):
    BLOCK_I = 16
    BLOCK_J = 16
    BLOCK_K = 16
    BLOCK_L = 16
    
    # First stage: compute tmp = alpha * A * B
    grid_1 = (triton.cdiv(NI, BLOCK_I), triton.cdiv(NJ, BLOCK_J))
    k2mm_kernel[grid_1](
        A.data_ptr(), B.data_ptr(), C.data_ptr(), D.data_ptr(), tmp.data_ptr(),
        alpha, beta,
        NI, NJ, NK, NL,
        BLOCK_I, BLOCK_J, BLOCK_K, BLOCK_L
    )
    
    # Second stage: compute D = beta * D + tmp * C
    grid_2 = (triton.cdiv(NI, BLOCK_I), triton.cdiv(NL, BLOCK_L))
    k2mm_second_stage_kernel[grid_2](
        tmp.data_ptr(), C.data_ptr(), D.data_ptr(), beta,
        NI, NJ, NL,
        BLOCK_I, BLOCK_L, BLOCK_J
    )