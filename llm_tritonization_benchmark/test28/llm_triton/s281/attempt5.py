import torch
import triton
import triton.language as tl

@triton.jit
def s281_expand_x_kernel(a_ptr, b_ptr, c_ptr, x_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    if pid != 0:
        return
    
    x_val = 0.0
    for i in range(n_elements):
        a_val = tl.load(a_ptr + (n_elements - 1 - i))
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        
        x_val = a_val + b_val * c_val
        
        tl.store(x_expanded_ptr + i, x_val)

@triton.jit
def s281_phase1_kernel(a_ptr, x_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    threshold = n_elements // 2
    
    mask = offsets < threshold
    
    x_vals = tl.load(x_expanded_ptr + offsets, mask=mask)
    
    tl.store(a_ptr + offsets, x_vals - 1.0, mask=mask)

@triton.jit
def s281_phase2_kernel(a_ptr, x_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    threshold = n_elements // 2
    block_start = threshold + pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    
    x_vals = tl.load(x_expanded_ptr + offsets, mask=mask)
    
    tl.store(a_ptr + offsets, x_vals - 1.0, mask=mask)

@triton.jit
def s281_update_b_kernel(b_ptr, x_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    
    x_vals = tl.load(x_expanded_ptr + offsets, mask=mask)
    
    tl.store(b_ptr + offsets, x_vals, mask=mask)

def s281_triton(a, b, c):
    n = a.shape[0]
    threshold = n // 2
    
    x_expanded = torch.zeros_like(a)
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    
    grid = (1,)
    s281_expand_x_kernel[grid](
        a_copy, b, c, x_expanded, n, BLOCK_SIZE
    )
    
    grid = (triton.cdiv(threshold, BLOCK_SIZE),)
    s281_phase1_kernel[grid](
        a, x_expanded, n, BLOCK_SIZE
    )
    
    remaining = n - threshold
    if remaining > 0:
        grid = (triton.cdiv(remaining, BLOCK_SIZE),)
        s281_phase2_kernel[grid](
            a, x_expanded, n, BLOCK_SIZE
        )
    
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    s281_update_b_kernel[grid](
        b, x_expanded, n, BLOCK_SIZE
    )