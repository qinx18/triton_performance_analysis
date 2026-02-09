import triton
import triton.language as tl
import torch

@triton.jit
def k2mm_kernel_first(A, B, tmp, alpha, NI, NJ, NK,
                      BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_J: tl.constexpr):
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
        
        a_ptrs = A + i_indices[:, None] * NK + k_indices[None, :]
        a_mask = i_mask[:, None] & k_mask[None, :]
        a_vals = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        b_ptrs = B + k_indices[:, None] * NJ + j_indices[None, :]
        b_mask = k_mask[:, None] & j_mask[None, :]
        b_vals = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        acc += tl.dot(a_vals, b_vals)
    
    tmp_vals = acc * alpha
    tmp_ptrs = tmp + i_indices[:, None] * NJ + j_indices[None, :]
    store_mask = i_mask[:, None] & j_mask[None, :]
    tl.store(tmp_ptrs, tmp_vals, mask=store_mask)

@triton.jit
def k2mm_kernel_second(tmp, C, D, beta, NI, NJ, NL,
                       BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_L: tl.constexpr):
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
    
    d_ptrs = D + i_indices[:, None] * NL + l_indices[None, :]
    d_mask = i_mask[:, None] & l_mask[None, :]
    d_vals = tl.load(d_ptrs, mask=d_mask, other=0.0)
    acc = d_vals * beta
    
    for j_start in range(0, NJ, 32):
        j_indices = j_start + j_offsets
        j_mask = j_indices < NJ
        
        tmp_ptrs = tmp + i_indices[:, None] * NJ + j_indices[None, :]
        tmp_mask = i_mask[:, None] & j_mask[None, :]
        tmp_vals = tl.load(tmp_ptrs, mask=tmp_mask, other=0.0)
        
        c_ptrs = C + j_indices[:, None] * NL + l_indices[None, :]
        c_mask = j_mask[:, None] & l_mask[None, :]
        c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0)
        
        acc += tl.dot(tmp_vals, c_vals)
    
    tl.store(d_ptrs, acc, mask=d_mask)

@triton.jit
def k2mm_kernel(A, B, C, D, tmp, alpha, beta, NI, NJ, NK, NL):
    pid = tl.program_id(0)
    
    if pid < NI * NJ:
        i = pid // NJ
        j = pid % NJ
        
        tmp_val = 0.0
        for k in range(NK):
            a_ptr = A + i * NK + k
            b_ptr = B + k * NJ + j
            a_val = tl.load(a_ptr)
            b_val = tl.load(b_ptr)
            tmp_val += alpha * a_val * b_val
        
        tmp_ptr = tmp + i * NJ + j
        tl.store(tmp_ptr, tmp_val)
    
    elif pid < NI * NJ + NI * NL:
        idx = pid - NI * NJ
        i = idx // NL
        l = idx % NL
        
        d_ptr = D + i * NL + l
        d_val = tl.load(d_ptr) * beta
        
        for j in range(NJ):
            tmp_ptr = tmp + i * NJ + j
            c_ptr = C + j * NL + l
            tmp_val = tl.load(tmp_ptr)
            c_val = tl.load(c_ptr)
            d_val += tmp_val * c_val
        
        tl.store(d_ptr, d_val)

def k2mm_triton(A, B, C, D, tmp, alpha, beta, NI, NJ, NK, NL):
    total_work = NI * NJ + NI * NL
    grid = (total_work,)
    
    k2mm_kernel[grid](A, B, C, D, tmp, alpha, beta, NI, NJ, NK, NL)