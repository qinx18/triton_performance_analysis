import triton
import triton.language as tl
import torch

@triton.jit
def s113_kernel(a_ptr, b_ptr, n):
    # Phase 1: Save original a[0] and update it
    if tl.program_id(0) == 0:
        orig_a0 = tl.load(a_ptr)
        b0 = tl.load(b_ptr)
        new_a0 = orig_a0 + b0
        tl.store(a_ptr, new_a0)

@triton.jit  
def s113_kernel_phase2(a_ptr, b_ptr, n):
    # Phase 2: Update a[1:] using the updated a[0]
    BLOCK_SIZE = 256
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE + 1  # Start from index 1
    
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n
    
    # Load updated a[0]
    a0_val = tl.load(a_ptr)
    
    # Load b values
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Compute and store results
    results = a0_val + b_vals
    tl.store(a_ptr + indices, results, mask=mask)

def s113_triton(a, b):
    n = a.shape[0]
    
    if n <= 1:
        return
    
    # Phase 1: Handle i = 0 (sequential)
    grid = (1,)
    s113_kernel[grid](a, b, n)
    
    # Phase 2: Handle i = 1 to n-1 (parallel)
    BLOCK_SIZE = 256
    remaining = n - 1
    grid = (triton.cdiv(remaining, BLOCK_SIZE),)
    s113_kernel_phase2[grid](a, b, n)