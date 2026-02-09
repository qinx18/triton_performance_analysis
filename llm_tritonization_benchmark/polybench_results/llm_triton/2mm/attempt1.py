import triton
import triton.language as tl
import torch

@triton.jit
def k2mm_kernel(A, B, C, D, tmp, alpha, beta, NI, NJ, NK, NL, 
                BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_J: tl.constexpr, BLOCK_SIZE_L: tl.constexpr):
    # First matrix multiplication: tmp = alpha * A * B
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    i_start = pid_i * BLOCK_SIZE_I
    j_start = pid_j * BLOCK_SIZE_J
    
    i_offsets = tl.arange(0, BLOCK_SIZE_I)
    j_offsets = tl.arange(0, BLOCK_SIZE_J)
    k_offsets = tl.arange(0, 32)  # Block size for reduction dimension
    
    i_indices = i_start + i_offsets
    j_indices = j_start + j_offsets
    
    i_mask = i_indices < NI
    j_mask = j_indices < NJ
    
    # Initialize accumulator for tmp
    acc = tl.zeros((BLOCK_SIZE_I, BLOCK_SIZE_J), dtype=tl.float32)
    
    # Compute tmp[i][j] = alpha * sum(A[i][k] * B[k][j])
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
        
        # Accumulate A[i,k] * B[k,j]
        acc += tl.dot(a_vals, b_vals)
    
    # Scale by alpha and store tmp
    tmp_vals = alpha * acc
    tmp_ptrs = tmp + i_indices[:, None] * NJ + j_indices[None, :]
    tmp_mask = i_mask[:, None] & j_mask[None, :]
    tl.store(tmp_ptrs, tmp_vals, mask=tmp_mask)

@triton.jit
def k2mm_kernel_second(tmp, C, D, beta, NI, NJ, NL,
                       BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_L: tl.constexpr):
    # Second matrix multiplication: D = tmp * C + beta * D
    pid_i = tl.program_id(0)
    pid_l = tl.program_id(1)
    
    i_start = pid_i * BLOCK_SIZE_I
    l_start = pid_l * BLOCK_SIZE_L
    
    i_offsets = tl.arange(0, BLOCK_SIZE_I)
    l_offsets = tl.arange(0, BLOCK_SIZE_L)
    j_offsets = tl.arange(0, 32)  # Block size for reduction dimension
    
    i_indices = i_start + i_offsets
    l_indices = l_start + l_offsets
    
    i_mask = i_indices < NI
    l_mask = l_indices < NL
    
    # Load and scale existing D values
    d_ptrs = D + i_indices[:, None] * NL + l_indices[None, :]
    d_mask = i_mask[:, None] & l_mask[None, :]
    d_vals = tl.load(d_ptrs, mask=d_mask, other=0.0)
    acc = beta * d_vals
    
    # Compute D[i][l] += sum(tmp[i][j] * C[j][l])
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
        
        # Accumulate tmp[i,j] * C[j,l]
        acc += tl.dot(tmp_vals, c_vals)
    
    # Store final result
    tl.store(d_ptrs, acc, mask=d_mask)

def k2mm_triton(A, B, C, D, tmp, alpha, beta, NI, NJ, NK, NL):
    BLOCK_SIZE_I = 16
    BLOCK_SIZE_J = 16
    BLOCK_SIZE_L = 16
    
    # First kernel: compute tmp = alpha * A * B
    grid1 = (triton.cdiv(NI, BLOCK_SIZE_I), triton.cdiv(NJ, BLOCK_SIZE_J))
    k2mm_kernel[grid1](
        A, B, C, D, tmp, alpha, beta, NI, NJ, NK, NL,
        BLOCK_SIZE_I=BLOCK_SIZE_I, BLOCK_SIZE_J=BLOCK_SIZE_J, BLOCK_SIZE_L=BLOCK_SIZE_L
    )
    
    # Second kernel: compute D = tmp * C + beta * D
    grid2 = (triton.cdiv(NI, BLOCK_SIZE_I), triton.cdiv(NL, BLOCK_SIZE_L))
    k2mm_kernel_second[grid2](
        tmp, C, D, beta, NI, NJ, NL,
        BLOCK_SIZE_I=BLOCK_SIZE_I, BLOCK_SIZE_L=BLOCK_SIZE_L
    )