import triton
import triton.language as tl
import torch

@triton.jit
def k3mm_kernel(A, B, C, D, E, F, G, NI, NJ, NK, NL, NM, 
                BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr,
                stage: tl.constexpr):
    
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    i_offsets = tl.arange(0, BLOCK_I)
    j_offsets = tl.arange(0, BLOCK_J)
    k_offsets = tl.arange(0, BLOCK_K)
    
    if stage == 0:
        # E := A*B
        i_indices = pid_i * BLOCK_I + i_offsets
        j_indices = pid_j * BLOCK_J + j_offsets
        
        i_mask = i_indices < NI
        j_mask = j_indices < NJ
        
        acc = tl.zeros((BLOCK_I, BLOCK_J), dtype=tl.float32)
        
        for k_start in range(0, NK, BLOCK_K):
            k_indices = k_start + k_offsets
            k_mask = k_indices < NK
            
            a_ptrs = A + i_indices[:, None] * NK + k_indices[None, :]
            a_mask = i_mask[:, None] & k_mask[None, :]
            a_vals = tl.load(a_ptrs, mask=a_mask, other=0.0)
            
            b_ptrs = B + k_indices[:, None] * NJ + j_indices[None, :]
            b_mask = k_mask[:, None] & j_mask[None, :]
            b_vals = tl.load(b_ptrs, mask=b_mask, other=0.0)
            
            acc += tl.dot(a_vals, b_vals)
        
        e_ptrs = E + i_indices[:, None] * NJ + j_indices[None, :]
        e_mask = i_mask[:, None] & j_mask[None, :]
        tl.store(e_ptrs, acc, mask=e_mask)
        
    elif stage == 1:
        # F := C*D
        i_indices = pid_i * BLOCK_I + i_offsets
        j_indices = pid_j * BLOCK_J + j_offsets
        
        i_mask = i_indices < NJ
        j_mask = j_indices < NL
        
        acc = tl.zeros((BLOCK_I, BLOCK_J), dtype=tl.float32)
        
        for k_start in range(0, NM, BLOCK_K):
            k_indices = k_start + k_offsets
            k_mask = k_indices < NM
            
            c_ptrs = C + i_indices[:, None] * NM + k_indices[None, :]
            c_mask = i_mask[:, None] & k_mask[None, :]
            c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0)
            
            d_ptrs = D + k_indices[:, None] * NL + j_indices[None, :]
            d_mask = k_mask[:, None] & j_mask[None, :]
            d_vals = tl.load(d_ptrs, mask=d_mask, other=0.0)
            
            acc += tl.dot(c_vals, d_vals)
        
        f_ptrs = F + i_indices[:, None] * NL + j_indices[None, :]
        f_mask = i_mask[:, None] & j_mask[None, :]
        tl.store(f_ptrs, acc, mask=f_mask)
        
    else:  # stage == 2
        # G := E*F
        i_indices = pid_i * BLOCK_I + i_offsets
        j_indices = pid_j * BLOCK_J + j_offsets
        
        i_mask = i_indices < NI
        j_mask = j_indices < NL
        
        acc = tl.zeros((BLOCK_I, BLOCK_J), dtype=tl.float32)
        
        for k_start in range(0, NJ, BLOCK_K):
            k_indices = k_start + k_offsets
            k_mask = k_indices < NJ
            
            e_ptrs = E + i_indices[:, None] * NJ + k_indices[None, :]
            e_mask = i_mask[:, None] & k_mask[None, :]
            e_vals = tl.load(e_ptrs, mask=e_mask, other=0.0)
            
            f_ptrs = F + k_indices[:, None] * NL + j_indices[None, :]
            f_mask = k_mask[:, None] & j_mask[None, :]
            f_vals = tl.load(f_ptrs, mask=f_mask, other=0.0)
            
            acc += tl.dot(e_vals, f_vals)
        
        g_ptrs = G + i_indices[:, None] * NL + j_indices[None, :]
        g_mask = i_mask[:, None] & j_mask[None, :]
        tl.store(g_ptrs, acc, mask=g_mask)

def k3mm_triton(A, B, C, D, E, F, G, NI, NJ, NK, NL, NM):
    BLOCK_I = 64
    BLOCK_J = 64  
    BLOCK_K = 32
    
    # Stage 0: E := A*B (NI x NJ)
    grid_0 = (triton.cdiv(NI, BLOCK_I), triton.cdiv(NJ, BLOCK_J))
    k3mm_kernel[grid_0](A, B, C, D, E, F, G, NI, NJ, NK, NL, NM, BLOCK_I, BLOCK_J, BLOCK_K, 0)
    
    # Stage 1: F := C*D (NJ x NL)
    grid_1 = (triton.cdiv(NJ, BLOCK_I), triton.cdiv(NL, BLOCK_J))
    k3mm_kernel[grid_1](A, B, C, D, E, F, G, NI, NJ, NK, NL, NM, BLOCK_I, BLOCK_J, BLOCK_K, 1)
    
    # Stage 2: G := E*F (NI x NL)
    grid_2 = (triton.cdiv(NI, BLOCK_I), triton.cdiv(NL, BLOCK_J))
    k3mm_kernel[grid_2](A, B, C, D, E, F, G, NI, NJ, NK, NL, NM, BLOCK_I, BLOCK_J, BLOCK_K, 2)