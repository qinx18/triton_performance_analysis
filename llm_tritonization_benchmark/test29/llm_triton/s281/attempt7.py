import triton
import triton.language as tl
import torch

@triton.jit
def s281_kernel_phase1(a_ptr, b_ptr, c_ptr, a_copy_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n // 2
    
    # Load b[i] and c[i]
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + indices, mask=mask, other=0.0)
    
    # Load a_copy[n-1-i]
    reverse_indices = n - 1 - indices
    a_reverse_vals = tl.load(a_copy_ptr + reverse_indices, mask=mask, other=0.0)
    
    # Compute x = a_copy[n-1-i] + b[i] * c[i]
    x_vals = a_reverse_vals + b_vals * c_vals
    
    # Store a[i] = x - 1.0
    tl.store(a_ptr + indices, x_vals - 1.0, mask=mask)
    
    # Store b[i] = x
    tl.store(b_ptr + indices, x_vals, mask=mask)

@triton.jit
def s281_kernel_phase2(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets + n // 2
    
    mask = indices < n
    
    # Load b[i] and c[i]
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + indices, mask=mask, other=0.0)
    
    # Load a[n-1-i] (updated values from phase 1)
    reverse_indices = n - 1 - indices
    a_reverse_vals = tl.load(a_ptr + reverse_indices, mask=mask, other=0.0)
    
    # Compute x = a[n-1-i] + b[i] * c[i]
    x_vals = a_reverse_vals + b_vals * c_vals
    
    # Store a[i] = x - 1.0
    tl.store(a_ptr + indices, x_vals - 1.0, mask=mask)
    
    # Store b[i] = x
    tl.store(b_ptr + indices, x_vals, mask=mask)

def s281_triton(a, b, c):
    n = a.shape[0]
    threshold = n // 2
    
    # Clone array for phase 1 reads
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    
    # Phase 1: process first half
    grid1 = (triton.cdiv(threshold, BLOCK_SIZE),)
    s281_kernel_phase1[grid1](a, b, c, a_copy, n, BLOCK_SIZE)
    
    # Phase 2: process second half
    remaining = n - threshold
    grid2 = (triton.cdiv(remaining, BLOCK_SIZE),)
    s281_kernel_phase2[grid2](a, b, c, n, BLOCK_SIZE)