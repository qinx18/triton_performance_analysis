import triton
import triton.language as tl
import torch

@triton.jit
def k3mm_kernel(A, B, C, D, E, F, G, NI, NJ, NK, NL, NM, 
                BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr):
    
    # E := A*B
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    i_offsets = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    j_offsets = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)
    
    i_mask = i_offsets < NI
    j_mask = j_offsets < NJ
    
    # Initialize accumulator for E[i,j]
    acc_e = tl.zeros((BLOCK_I, BLOCK_J), dtype=tl.float32)
    
    k_offsets = tl.arange(0, BLOCK_K)
    for k_start in range(0, NK, BLOCK_K):
        k_current = k_start + k_offsets
        k_mask = k_current < NK
        
        # Load A[i, k]
        a_ptrs = A + i_offsets[:, None] * NK + k_current[None, :]
        a_mask = i_mask[:, None] & k_mask[None, :]
        a_vals = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # Load B[k, j]
        b_ptrs = B + k_current[:, None] * NJ + j_offsets[None, :]
        b_mask = k_mask[:, None] & j_mask[None, :]
        b_vals = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Accumulate
        acc_e += tl.dot(a_vals, b_vals)
    
    # Store E[i, j]
    e_ptrs = E + i_offsets[:, None] * NJ + j_offsets[None, :]
    e_mask = i_mask[:, None] & j_mask[None, :]
    tl.store(e_ptrs, acc_e, mask=e_mask)

@triton.jit
def k3mm_kernel_f(C, D, F, NJ, NL, NM,
                  BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr):
    
    # F := C*D
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    i_offsets = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    j_offsets = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)
    
    i_mask = i_offsets < NJ
    j_mask = j_offsets < NL
    
    # Initialize accumulator for F[i,j]
    acc_f = tl.zeros((BLOCK_I, BLOCK_J), dtype=tl.float32)
    
    k_offsets = tl.arange(0, BLOCK_K)
    for k_start in range(0, NM, BLOCK_K):
        k_current = k_start + k_offsets
        k_mask = k_current < NM
        
        # Load C[i, k]
        c_ptrs = C + i_offsets[:, None] * NM + k_current[None, :]
        c_mask = i_mask[:, None] & k_mask[None, :]
        c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0)
        
        # Load D[k, j]
        d_ptrs = D + k_current[:, None] * NL + j_offsets[None, :]
        d_mask = k_mask[:, None] & j_mask[None, :]
        d_vals = tl.load(d_ptrs, mask=d_mask, other=0.0)
        
        # Accumulate
        acc_f += tl.dot(c_vals, d_vals)
    
    # Store F[i, j]
    f_ptrs = F + i_offsets[:, None] * NL + j_offsets[None, :]
    f_mask = i_mask[:, None] & j_mask[None, :]
    tl.store(f_ptrs, acc_f, mask=f_mask)

@triton.jit
def k3mm_kernel_g(E, F, G, NI, NJ, NL,
                  BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr):
    
    # G := E*F
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    i_offsets = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    j_offsets = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)
    
    i_mask = i_offsets < NI
    j_mask = j_offsets < NL
    
    # Initialize accumulator for G[i,j]
    acc_g = tl.zeros((BLOCK_I, BLOCK_J), dtype=tl.float32)
    
    k_offsets = tl.arange(0, BLOCK_K)
    for k_start in range(0, NJ, BLOCK_K):
        k_current = k_start + k_offsets
        k_mask = k_current < NJ
        
        # Load E[i, k]
        e_ptrs = E + i_offsets[:, None] * NJ + k_current[None, :]
        e_mask = i_mask[:, None] & k_mask[None, :]
        e_vals = tl.load(e_ptrs, mask=e_mask, other=0.0)
        
        # Load F[k, j]
        f_ptrs = F + k_current[:, None] * NL + j_offsets[None, :]
        f_mask = k_mask[:, None] & j_mask[None, :]
        f_vals = tl.load(f_ptrs, mask=f_mask, other=0.0)
        
        # Accumulate
        acc_g += tl.dot(e_vals, f_vals)
    
    # Store G[i, j]
    g_ptrs = G + i_offsets[:, None] * NL + j_offsets[None, :]
    g_mask = i_mask[:, None] & j_mask[None, :]
    tl.store(g_ptrs, acc_g, mask=g_mask)

def k3mm_triton(A, B, C, D, E, F, G, NI, NJ, NK, NL, NM):
    BLOCK_I = 16
    BLOCK_J = 16
    BLOCK_K = 32
    
    # E := A*B
    grid_e = (triton.cdiv(NI, BLOCK_I), triton.cdiv(NJ, BLOCK_J))
    k3mm_kernel[grid_e](A, B, C, D, E, F, G, NI, NJ, NK, NL, NM, BLOCK_I, BLOCK_J, BLOCK_K)
    
    # F := C*D
    grid_f = (triton.cdiv(NJ, BLOCK_I), triton.cdiv(NL, BLOCK_J))
    k3mm_kernel_f[grid_f](C, D, F, NJ, NL, NM, BLOCK_I, BLOCK_J, BLOCK_K)
    
    # G := E*F
    grid_g = (triton.cdiv(NI, BLOCK_I), triton.cdiv(NL, BLOCK_J))
    k3mm_kernel_g[grid_g](E, F, G, NI, NJ, NL, BLOCK_I, BLOCK_J, BLOCK_K)