import torch
import triton
import triton.language as tl

@triton.jit
def s281_expand_x_kernel(a_ptr, b_ptr, c_ptr, x_expanded_ptr, n):
    if tl.program_id(0) != 0:
        return
    
    for i in range(n):
        a_val = tl.load(a_ptr + (n - 1 - i))
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        
        x_val = a_val + b_val * c_val
        tl.store(x_expanded_ptr + i, x_val)

@triton.jit
def s281_phase1_kernel(a_ptr, b_ptr, x_expanded_ptr, n, threshold, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    block_start = pid * BLOCK_SIZE
    indices = block_start + offsets
    mask = indices < threshold
    
    x_vals = tl.load(x_expanded_ptr + indices, mask=mask)
    a_vals = x_vals - 1.0
    tl.store(a_ptr + indices, a_vals, mask=mask)
    tl.store(b_ptr + indices, x_vals, mask=mask)

@triton.jit
def s281_phase2_kernel(a_ptr, b_ptr, x_expanded_ptr, n, threshold, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    block_start = pid * BLOCK_SIZE + threshold
    indices = block_start + offsets
    mask = indices < n
    
    x_vals = tl.load(x_expanded_ptr + indices, mask=mask)
    a_vals = x_vals - 1.0
    tl.store(a_ptr + indices, a_vals, mask=mask)
    tl.store(b_ptr + indices, x_vals, mask=mask)

def s281_triton(a, b, c):
    n = a.shape[0]
    threshold = n // 2
    
    a_copy = a.clone()
    x_expanded = torch.zeros(n, dtype=a.dtype, device=a.device)
    
    s281_expand_x_kernel[(1,)](a_copy, b, c, x_expanded, n)
    
    BLOCK_SIZE = 256
    
    if threshold > 0:
        num_blocks_phase1 = triton.cdiv(threshold, BLOCK_SIZE)
        s281_phase1_kernel[(num_blocks_phase1,)](a, b, x_expanded, n, threshold, BLOCK_SIZE)
    
    remaining = n - threshold
    if remaining > 0:
        num_blocks_phase2 = triton.cdiv(remaining, BLOCK_SIZE)
        s281_phase2_kernel[(num_blocks_phase2,)](a, b, x_expanded, n, threshold, BLOCK_SIZE)