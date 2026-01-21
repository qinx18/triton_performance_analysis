import torch
import triton
import triton.language as tl

@triton.jit
def s281_kernel(a_ptr, b_ptr, c_ptr, n, threshold, phase, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    if phase == 1:
        # Phase 1: i = 0 to threshold-1
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < threshold
        
        # Read indices for a: n-1-i
        read_offsets = n - 1 - offsets
        read_mask = mask & (read_offsets >= 0) & (read_offsets < n)
        
        a_vals = tl.load(a_ptr + read_offsets, mask=read_mask)
        b_vals = tl.load(b_ptr + offsets, mask=mask)
        c_vals = tl.load(c_ptr + offsets, mask=mask)
        
        x_vals = a_vals + b_vals * c_vals
        
        tl.store(a_ptr + offsets, x_vals - 1.0, mask=mask)
        tl.store(b_ptr + offsets, x_vals, mask=mask)
        
    else:
        # Phase 2: i = threshold to n-1
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) + threshold
        mask = offsets < n
        
        # Read indices for a: n-1-i
        read_offsets = n - 1 - offsets
        read_mask = mask & (read_offsets >= 0) & (read_offsets < n)
        
        a_vals = tl.load(a_ptr + read_offsets, mask=read_mask)
        b_vals = tl.load(b_ptr + offsets, mask=mask)
        c_vals = tl.load(c_ptr + offsets, mask=mask)
        
        x_vals = a_vals + b_vals * c_vals
        
        tl.store(a_ptr + offsets, x_vals - 1.0, mask=mask)
        tl.store(b_ptr + offsets, x_vals, mask=mask)

def s281_triton(a, b, c, x):
    n = a.shape[0]
    threshold = n // 2
    BLOCK_SIZE = 256
    
    # Phase 1: i = 0 to threshold-1
    grid1 = (triton.cdiv(threshold, BLOCK_SIZE),)
    s281_kernel[grid1](a, b, c, n, threshold, 1, BLOCK_SIZE=BLOCK_SIZE)
    
    # Phase 2: i = threshold to n-1
    remaining = n - threshold
    if remaining > 0:
        grid2 = (triton.cdiv(remaining, BLOCK_SIZE),)
        s281_kernel[grid2](a, b, c, n, threshold, 2, BLOCK_SIZE=BLOCK_SIZE)