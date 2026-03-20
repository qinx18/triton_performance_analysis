import triton
import triton.language as tl
import torch

@triton.jit
def k3mm_kernel_phase1(A_ptr, B_ptr, E_ptr, NI: tl.constexpr, NJ: tl.constexpr, NK: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_start in range(0, NK, BLOCK_K):
        k_current = k_start + k_offsets
        
        a_ptrs = A_ptr + m_offsets[:, None] * NK + k_current[None, :]
        b_ptrs = B_ptr + k_current[:, None] * NJ + n_offsets[None, :]
        
        a_mask = (m_offsets[:, None] < NI) & (k_current[None, :] < NK)
        b_mask = (k_current[:, None] < NK) & (n_offsets[None, :] < NJ)
        
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        acc += tl.dot(a, b)
    
    e_ptrs = E_ptr + m_offsets[:, None] * NJ + n_offsets[None, :]
    e_mask = (m_offsets[:, None] < NI) & (n_offsets[None, :] < NJ)
    
    tl.store(e_ptrs, acc, mask=e_mask)

@triton.jit
def k3mm_kernel_phase2(C_ptr, D_ptr, F_ptr, NJ: tl.constexpr, NL: tl.constexpr, NM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_start in range(0, NM, BLOCK_K):
        k_current = k_start + k_offsets
        
        c_ptrs = C_ptr + m_offsets[:, None] * NM + k_current[None, :]
        d_ptrs = D_ptr + k_current[:, None] * NL + n_offsets[None, :]
        
        c_mask = (m_offsets[:, None] < NJ) & (k_current[None, :] < NM)
        d_mask = (k_current[:, None] < NM) & (n_offsets[None, :] < NL)
        
        c = tl.load(c_ptrs, mask=c_mask, other=0.0)
        d = tl.load(d_ptrs, mask=d_mask, other=0.0)
        
        acc += tl.dot(c, d)
    
    f_ptrs = F_ptr + m_offsets[:, None] * NL + n_offsets[None, :]
    f_mask = (m_offsets[:, None] < NJ) & (n_offsets[None, :] < NL)
    
    tl.store(f_ptrs, acc, mask=f_mask)

@triton.jit
def k3mm_kernel_phase3(E_ptr, F_ptr, G_ptr, NI: tl.constexpr, NJ: tl.constexpr, NL: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_start in range(0, NJ, BLOCK_K):
        k_current = k_start + k_offsets
        
        e_ptrs = E_ptr + m_offsets[:, None] * NJ + k_current[None, :]
        f_ptrs = F_ptr + k_current[:, None] * NL + n_offsets[None, :]
        
        e_mask = (m_offsets[:, None] < NI) & (k_current[None, :] < NJ)
        f_mask = (k_current[:, None] < NJ) & (n_offsets[None, :] < NL)
        
        e = tl.load(e_ptrs, mask=e_mask, other=0.0)
        f = tl.load(f_ptrs, mask=f_mask, other=0.0)
        
        acc += tl.dot(e, f)
    
    g_ptrs = G_ptr + m_offsets[:, None] * NL + n_offsets[None, :]
    g_mask = (m_offsets[:, None] < NI) & (n_offsets[None, :] < NL)
    
    tl.store(g_ptrs, acc, mask=g_mask)

def k3mm_triton(A, B, C, D, E, F, G, NI, NJ, NK, NL, NM):
    BLOCK_SIZE = 16
    
    # Phase 1: E := A*B (NI x NJ)
    grid1 = (triton.cdiv(NI, BLOCK_SIZE), triton.cdiv(NJ, BLOCK_SIZE))
    k3mm_kernel_phase1[grid1](A, B, E, NI, NJ, NK, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
    
    # Phase 2: F := C*D (NJ x NL)
    grid2 = (triton.cdiv(NJ, BLOCK_SIZE), triton.cdiv(NL, BLOCK_SIZE))
    k3mm_kernel_phase2[grid2](C, D, F, NJ, NL, NM, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
    
    # Phase 3: G := E*F (NI x NL)
    grid3 = (triton.cdiv(NI, BLOCK_SIZE), triton.cdiv(NL, BLOCK_SIZE))
    k3mm_kernel_phase3[grid3](E, F, G, NI, NJ, NL, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)