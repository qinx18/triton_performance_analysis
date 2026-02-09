import triton
import triton.language as tl
import torch

@triton.jit
def k3mm_kernel(A, B, C, D, E, F, G, NI, NJ, NK, NL, NM, 
                BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr):
    
    # E := A*B
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    pid_stage = tl.program_id(2)
    
    if pid_stage == 0:
        # E := A*B
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
        
    elif pid_stage == 1:
        # F := C*D (reuse i,j for NJ,NL dimensions)
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
        
    elif pid_stage == 2:
        # G := E*F (reuse i,j for NI,NL dimensions)
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
    
    # Stage 0: E := A*B
    max_i_0 = triton.cdiv(NI, BLOCK_I)
    max_j_0 = triton.cdiv(NJ, BLOCK_J)
    grid_0 = (max_i_0, max_j_0, 1)
    k3mm_kernel[grid_0](A, B, C, D, E, F, G, NI, NJ, NK, NL, NM, BLOCK_I, BLOCK_J, BLOCK_K)
    
    # Stage 1: F := C*D
    max_i_1 = triton.cdiv(NJ, BLOCK_I)
    max_j_1 = triton.cdiv(NL, BLOCK_J)
    grid_1 = (max_i_1, max_j_1, 2)
    k3mm_kernel[grid_1](A, B, C, D, E, F, G, NI, NJ, NK, NL, NM, BLOCK_I, BLOCK_J, BLOCK_K)
    
    # Stage 2: G := E*F
    max_i_2 = triton.cdiv(NI, BLOCK_I)
    max_j_2 = triton.cdiv(NL, BLOCK_J)
    grid_2 = (max_i_2, max_j_2, 3)
    k3mm_kernel[grid_2](A, B, C, D, E, F, G, NI, NJ, NK, NL, NM, BLOCK_I, BLOCK_J, BLOCK_K)