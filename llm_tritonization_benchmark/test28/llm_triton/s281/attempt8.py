import triton
import triton.language as tl
import torch

@triton.jit
def s281_phase1_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, n, threshold, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < threshold
    
    # Load b and c values
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + indices, mask=mask, other=0.0)
    
    # Load a values from reverse indices using original copy
    reverse_indices = n - 1 - indices
    a_reverse_vals = tl.load(a_copy_ptr + reverse_indices, mask=mask, other=0.0)
    
    # Compute x = a[n-1-i] + b[i] * c[i]
    x_vals = a_reverse_vals + b_vals * c_vals
    
    # Store a[i] = x - 1.0
    a_new_vals = x_vals - 1.0
    tl.store(a_ptr + indices, a_new_vals, mask=mask)
    
    # Store b[i] = x
    tl.store(b_ptr + indices, x_vals, mask=mask)

@triton.jit
def s281_phase2_kernel(a_ptr, b_ptr, c_ptr, n, threshold, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE + threshold
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n
    
    # Load b and c values
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + indices, mask=mask, other=0.0)
    
    # Load a values from reverse indices (now updated from phase 1)
    reverse_indices = n - 1 - indices
    a_reverse_vals = tl.load(a_ptr + reverse_indices, mask=mask, other=0.0)
    
    # Compute x = a[n-1-i] + b[i] * c[i]
    x_vals = a_reverse_vals + b_vals * c_vals
    
    # Store a[i] = x - 1.0
    a_new_vals = x_vals - 1.0
    tl.store(a_ptr + indices, a_new_vals, mask=mask)
    
    # Store b[i] = x
    tl.store(b_ptr + indices, x_vals, mask=mask)

def s281_triton(a, b, c):
    n = a.shape[0]
    threshold = n // 2
    
    # Clone array for Phase 1 reads
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    
    # Phase 1: i = 0 to threshold-1
    grid_size_1 = triton.cdiv(threshold, BLOCK_SIZE)
    if grid_size_1 > 0:
        s281_phase1_kernel[(grid_size_1,)](a, a_copy, b, c, n, threshold, BLOCK_SIZE)
    
    # Phase 2: i = threshold to n-1
    remaining = n - threshold
    grid_size_2 = triton.cdiv(remaining, BLOCK_SIZE)
    if grid_size_2 > 0:
        s281_phase2_kernel[(grid_size_2,)](a, b, c, n, threshold, BLOCK_SIZE)