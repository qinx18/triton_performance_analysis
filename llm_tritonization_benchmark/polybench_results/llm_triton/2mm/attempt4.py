import triton
import triton.language as tl
import torch

@triton.jit
def k2mm_kernel_first(A, B, tmp, alpha, NI, NJ, NK,
                      BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_J: tl.constexpr):
    # First stage: tmp[i][j] = alpha * sum(A[i][k] * B[k][j])
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    i_start = pid_i * BLOCK_SIZE_I
    j_start = pid_j * BLOCK_SIZE_J
    
    i_offsets = tl.arange(0, BLOCK_SIZE_I)
    j_offsets = tl.arange(0, BLOCK_SIZE_J)
    k_offsets = tl.arange(0, 32)
    
    i_indices = i_start + i_offsets
    j_indices = j_start + j_offsets
    
    i_mask = i_indices < NI
    j_mask = j_indices < NJ
    
    acc = tl.zeros((BLOCK_SIZE_I, BLOCK_SIZE_J), dtype=tl.float32)
    
    for k_start in range(0, NK, 32):
        k_indices = k_start + k_offsets
        k_mask = k_indices < NK
        
        # Load A[i, k]
        a_ptrs = A + i_indices[:, None] * NK + k_indices[None, :]
        a_mask = i_mask[:, None] & k_mask[None, :]
        a_vals = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # Load B[k, j]
        b_ptrs = B + k_indices[:, None] * NJ + j_indices[None, :]
        b_mask = k_mask[:, None] & j_mask[None, :]
        b_vals = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        acc += tl.dot(a_vals, b_vals)
    
    # Store tmp with alpha scaling (alpha moved inside acc)
    tmp_vals = acc * alpha
    tmp_ptrs = tmp + i_indices[:, None] * NJ + j_indices[None, :]
    store_mask = i_mask[:, None] & j_mask[None, :]
    tl.store(tmp_ptrs, tmp_vals, mask=store_mask)

@triton.jit
def k2mm_kernel_second(tmp, C, D, beta, NI, NJ, NL,
                       BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_L: tl.constexpr):
    # Second stage: D[i][l] = beta * D[i][l] + sum(tmp[i][j] * C[j][l])
    pid_i = tl.program_id(0)
    pid_l = tl.program_id(1)
    
    i_start = pid_i * BLOCK_SIZE_I
    l_start = pid_l * BLOCK_SIZE_L
    
    i_offsets = tl.arange(0, BLOCK_SIZE_I)
    l_offsets = tl.arange(0, BLOCK_SIZE_L)
    j_offsets = tl.arange(0, 32)
    
    i_indices = i_start + i_offsets
    l_indices = l_start + l_offsets
    
    i_mask = i_indices < NI
    l_mask = l_indices < NL
    
    # Load and scale D by beta
    d_ptrs = D + i_indices[:, None] * NL + l_indices[None, :]
    d_mask = i_mask[:, None] & l_mask[None, :]
    d_vals = tl.load(d_ptrs, mask=d_mask, other=0.0)
    acc = d_vals * beta
    
    # Accumulate tmp * C
    for j_start in range(0, NJ, 32):
        j_indices = j_start + j_offsets
        j_mask = j_indices < NJ
        
        # Load tmp[i, j]
        tmp_ptrs = tmp + i_indices[:, None] * NJ + j_indices[None, :]
        tmp_mask = i_mask[:, None] & j_mask[None, :]
        tmp_vals = tl.load(tmp_ptrs, mask=tmp_mask, other=0.0)
        
        # Load C[j, l]
        c_ptrs = C + j_indices[:, None] * NL + l_indices[None, :]
        c_mask = j_mask[:, None] & l_mask[None, :]
        c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0)
        
        acc += tl.dot(tmp_vals, c_vals)
    
    # Store result
    tl.store(d_ptrs, acc, mask=d_mask)

def k2mm_triton(A, B, C, D, tmp, alpha, beta, NI, NJ, NK, NL):
    BLOCK_SIZE_I = 16
    BLOCK_SIZE_J = 16
    BLOCK_SIZE_L = 16
    
    # First stage: compute tmp = alpha * A * B
    grid1 = (triton.cdiv(NI, BLOCK_SIZE_I), triton.cdiv(NJ, BLOCK_SIZE_J))
    k2mm_kernel_first[grid1](
        A, B, tmp, alpha, NI, NJ, NK,
        BLOCK_SIZE_I=BLOCK_SIZE_I, BLOCK_SIZE_J=BLOCK_SIZE_J
    )
    
    # Second stage: compute D = tmp * C + beta * D
    grid2 = (triton.cdiv(NI, BLOCK_SIZE_I), triton.cdiv(NL, BLOCK_SIZE_L))
    k2mm_kernel_second[grid2](
        tmp, C, D, beta, NI, NJ, NL,
        BLOCK_SIZE_I=BLOCK_SIZE_I, BLOCK_SIZE_L=BLOCK_SIZE_L
    )