import torch
import triton
import triton.language as tl

@triton.jit
def s291_expand_im1_kernel(im1_expanded_ptr, n):
    # Single thread expands the scalar im1 variable
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    im1_val = n - 1  # Initial value: LEN_1D - 1
    for i in range(n):
        tl.store(im1_expanded_ptr + i, im1_val)
        im1_val = i  # Update for next iteration

@triton.jit
def s291_kernel(a_ptr, b_ptr, im1_expanded_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # Load b values at current positions
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Load im1 values (previous indices)
    im1_vals = tl.load(im1_expanded_ptr + offsets, mask=mask)
    
    # Load b values at previous indices
    b_prev_vals = tl.load(b_ptr + im1_vals, mask=mask)
    
    # Compute result
    result = (b_vals + b_prev_vals) * 0.5
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s291_triton(a, b):
    n = a.shape[0]
    
    # Create expanded im1 array
    im1_expanded = torch.empty(n, dtype=torch.int32, device=a.device)
    
    # Expand the scalar im1 variable
    grid = (1,)
    s291_expand_im1_kernel[grid](im1_expanded, n)
    
    # Compute the main kernel
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    s291_kernel[grid](a, b, im1_expanded, n, BLOCK_SIZE)