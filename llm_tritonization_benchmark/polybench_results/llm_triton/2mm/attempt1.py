import triton
import triton.language as tl
import torch

@triton.jit
def k2mm_phase1_kernel(A_ptr, B_ptr, tmp_ptr, alpha, 
                       NI: tl.constexpr, NJ: tl.constexpr, NK: tl.constexpr,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_start in range(0, NK, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        
        # Load A tile
        a_ptrs = A_ptr + m_offsets[:, None] * NK + k_offsets[None, :]
        a_mask = (m_offsets[:, None] < NI) & (k_offsets[None, :] < NK)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # Load B tile
        b_ptrs = B_ptr + k_offsets[:, None] * NJ + n_offsets[None, :]
        b_mask = (k_offsets[:, None] < NK) & (n_offsets[None, :] < NJ)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        acc += tl.dot(a, b)
    
    # Scale by alpha and store tmp
    acc = alpha * acc
    tmp_ptrs = tmp_ptr + m_offsets[:, None] * NJ + n_offsets[None, :]
    tmp_mask = (m_offsets[:, None] < NI) & (n_offsets[None, :] < NJ)
    tl.store(tmp_ptrs, acc, mask=tmp_mask)

@triton.jit
def k2mm_phase2_kernel(tmp_ptr, C_ptr, D_ptr, beta,
                       NI: tl.constexpr, NJ: tl.constexpr, NL: tl.constexpr,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    
    # Load and scale D by beta
    d_ptrs = D_ptr + m_offsets[:, None] * NL + n_offsets[None, :]
    d_mask = (m_offsets[:, None] < NI) & (n_offsets[None, :] < NL)
    acc = tl.load(d_ptrs, mask=d_mask, other=0.0) * beta
    
    for k_start in range(0, NJ, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        
        # Load tmp tile
        tmp_ptrs = tmp_ptr + m_offsets[:, None] * NJ + k_offsets[None, :]
        tmp_mask = (m_offsets[:, None] < NI) & (k_offsets[None, :] < NJ)
        tmp_tile = tl.load(tmp_ptrs, mask=tmp_mask, other=0.0)
        
        # Load C tile
        c_ptrs = C_ptr + k_offsets[:, None] * NL + n_offsets[None, :]
        c_mask = (k_offsets[:, None] < NJ) & (n_offsets[None, :] < NL)
        c_tile = tl.load(c_ptrs, mask=c_mask, other=0.0)
        
        acc += tl.dot(tmp_tile, c_tile)
    
    # Store result back to D
    tl.store(d_ptrs, acc, mask=d_mask)

def k2mm_triton(A, B, C, D, tmp, alpha, beta, NI, NJ, NK, NL):
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 16
    
    # Phase 1: tmp = alpha * A * B
    grid1 = (triton.cdiv(NI, BLOCK_M), triton.cdiv(NJ, BLOCK_N))
    k2mm_phase1_kernel[grid1](
        A, B, tmp, alpha,
        NI, NJ, NK,
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    
    # Phase 2: D = beta * D + tmp * C
    grid2 = (triton.cdiv(NI, BLOCK_M), triton.cdiv(NL, BLOCK_N))
    k2mm_phase2_kernel[grid2](
        tmp, C, D, beta,
        NI, NJ, NL,
        BLOCK_M, BLOCK_N, BLOCK_K
    )

def k2mm_kernel(A, B, C, D, tmp, alpha, beta, NI, NJ, NK, NL):
    k2mm_triton(A, B, C, D, tmp, alpha, beta, NI, NJ, NK, NL)