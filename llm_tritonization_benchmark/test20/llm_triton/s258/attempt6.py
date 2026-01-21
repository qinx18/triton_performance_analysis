import torch
import triton
import triton.language as tl

@triton.jit
def s258_expand_s_kernel(a_ptr, d_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    if block_start >= n_elements:
        return
    
    block_end = min(block_start + BLOCK_SIZE, n_elements)
    
    s_val = 0.0
    if pid > 0:
        s_val = tl.load(s_expanded_ptr + block_start - 1)
    
    for i in range(block_start, block_end):
        a_val = tl.load(a_ptr + i)
        if a_val > 0.0:
            d_val = tl.load(d_ptr + i)
            s_val = d_val * d_val
        tl.store(s_expanded_ptr + i, s_val)

@triton.jit
def s258_compute_kernel(s_expanded_ptr, c_ptr, d_ptr, aa_ptr, b_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    
    s_vals = tl.load(s_expanded_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    aa_vals = tl.load(aa_ptr + offsets, mask=mask)
    
    b_vals = s_vals * c_vals + d_vals
    e_vals = (s_vals + 1.0) * aa_vals
    
    tl.store(b_ptr + offsets, b_vals, mask=mask)
    tl.store(e_ptr + offsets, e_vals, mask=mask)

@triton.jit
def s258_sync_kernel(s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    if pid == 0:
        return
    
    block_start = pid * BLOCK_SIZE
    prev_block_end = block_start - 1
    
    if prev_block_end < n_elements:
        prev_s_val = tl.load(s_expanded_ptr + prev_block_end)
        
        for i in range(block_start, min(block_start + BLOCK_SIZE, n_elements)):
            current_s = tl.load(s_expanded_ptr + i)
            if i == block_start:
                if current_s == tl.load(s_expanded_ptr + block_start):
                    for j in range(block_start, min(block_start + BLOCK_SIZE, n_elements)):
                        tl.store(s_expanded_ptr + j, prev_s_val)
                        if j + 1 < n_elements:
                            next_val = tl.load(s_expanded_ptr + j + 1) 
                            if next_val != prev_s_val:
                                prev_s_val = next_val

def s258_triton(a, aa, b, c, d, e):
    n_elements = aa.shape[0]
    BLOCK_SIZE = 256
    
    s_expanded = torch.zeros(n_elements, dtype=torch.float32, device=a.device)
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s258_expand_s_kernel[grid](a, d, s_expanded, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    for _ in range(10):
        s258_sync_kernel[grid](s_expanded, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    s258_compute_kernel[grid](s_expanded, c, d, aa, b, e, n_elements, BLOCK_SIZE=BLOCK_SIZE)