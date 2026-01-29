import torch
import triton
import triton.language as tl

@triton.jit
def s172_kernel(a_ptr, b_ptr, n1, n3, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Each program processes one block of consecutive valid indices
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    block_indices = block_start + offsets
    
    # Convert block position back to actual array indices
    actual_indices = (n1 - 1) + block_indices * n3
    
    # Calculate number of valid elements
    max_elements = (N - n1 + n3) // n3
    
    # Mask for valid indices
    mask = (block_indices < max_elements) & (actual_indices < N)
    
    # Load, compute, and store
    a_vals = tl.load(a_ptr + actual_indices, mask=mask)
    b_vals = tl.load(b_ptr + actual_indices, mask=mask)
    result = a_vals + b_vals
    tl.store(a_ptr + actual_indices, result, mask=mask)

def s172_triton(a, b):
    N = a.shape[0]
    n1 = 1
    n3 = 2
    
    # Calculate total number of valid iterations
    if n1 - 1 >= N:
        return
    
    num_iterations = (N - (n1 - 1) + n3 - 1) // n3
    
    if num_iterations <= 0:
        return
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(num_iterations, BLOCK_SIZE),)
    
    s172_kernel[grid](a, b, n1, n3, N, BLOCK_SIZE)