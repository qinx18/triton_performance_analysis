import triton
import triton.language as tl
import torch

@triton.jit
def s1113_kernel_phase1(a_ptr, a_copy_ptr, b_ptr, threshold, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = (indices < threshold + 1) & (indices < n_elements)
    
    # Load the constant value from the copy
    a_val = tl.load(a_copy_ptr + threshold)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    result = a_val + b_vals
    tl.store(a_ptr + indices, result, mask=mask)

@triton.jit
def s1113_kernel_phase2(a_ptr, b_ptr, threshold, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets + threshold + 1
    
    mask = indices < n_elements
    
    # Load the updated value from the original array
    a_val = tl.load(a_ptr + threshold)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    result = a_val + b_vals
    tl.store(a_ptr + indices, result, mask=mask)

def s1113_triton(a, b):
    n = a.shape[0]
    threshold = n // 2
    
    # Create read-only copy for phase 1
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    
    # Phase 1: i = 0 to threshold (uses original value)
    num_elements_phase1 = threshold + 1
    grid_phase1 = (triton.cdiv(num_elements_phase1, BLOCK_SIZE),)
    s1113_kernel_phase1[grid_phase1](
        a, a_copy, b, threshold, n, BLOCK_SIZE
    )
    
    # Phase 2: i = threshold+1 to end (uses updated value)
    if threshold + 1 < n:
        num_elements_phase2 = n - threshold - 1
        grid_phase2 = (triton.cdiv(num_elements_phase2, BLOCK_SIZE),)
        s1113_kernel_phase2[grid_phase2](
            a, b, threshold, n, BLOCK_SIZE
        )