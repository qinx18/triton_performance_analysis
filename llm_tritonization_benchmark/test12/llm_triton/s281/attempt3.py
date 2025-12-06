import triton
import triton.language as tl
import torch

@triton.jit
def s281_kernel_phase1(a_ptr, a_copy_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE) + block_start
    mask = offsets < n
    
    # Load inputs
    reverse_offsets = (n - 1) - offsets
    
    a_reverse = tl.load(a_copy_ptr + reverse_offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    
    # Compute
    x = a_reverse + b_vals * c_vals
    a_new = x - 1.0
    
    # Store results
    tl.store(a_ptr + offsets, a_new, mask=mask)
    tl.store(b_ptr + offsets, x, mask=mask)

@triton.jit
def s281_kernel_phase2(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE) + block_start
    mask = offsets < n
    
    # Load inputs
    reverse_offsets = (n - 1) - offsets
    
    a_reverse = tl.load(a_ptr + reverse_offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    
    # Compute
    x = a_reverse + b_vals * c_vals
    a_new = x - 1.0
    
    # Store results
    tl.store(a_ptr + offsets, a_new, mask=mask)
    tl.store(b_ptr + offsets, x, mask=mask)

def s281_triton(a, b, c):
    n = a.shape[0]
    threshold = n // 2
    
    # Clone array for Phase 1 reads
    a_copy = a.clone()
    
    # Phase 1: i = 0 to threshold-1
    BLOCK_SIZE = 256
    if threshold > 0:
        grid1 = (triton.cdiv(threshold, BLOCK_SIZE),)
        s281_kernel_phase1[grid1](a, a_copy, b, c, threshold, BLOCK_SIZE)
    
    # Phase 2: i = threshold to end
    remaining = n - threshold
    if remaining > 0:
        BLOCK_SIZE = 256
        grid2 = (triton.cdiv(remaining, BLOCK_SIZE),)
        s281_kernel_phase2[grid2](a[threshold:], b[threshold:], c[threshold:], remaining, BLOCK_SIZE)