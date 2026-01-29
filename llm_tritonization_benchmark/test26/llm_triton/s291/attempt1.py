import triton
import triton.language as tl
import torch

@triton.jit
def s291_expand_im1_kernel(im1_expanded_ptr, n_elements):
    # Single thread computes the expanded im1 array
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    im1_val = n_elements - 1  # Initial value: LEN_1D-1
    for i in range(n_elements):
        tl.store(im1_expanded_ptr + i, im1_val)
        im1_val = i  # Update for next iteration

@triton.jit
def s291_kernel(a_ptr, b_ptr, im1_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load b values and im1 indices
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    im1_indices = tl.load(im1_expanded_ptr + offsets, mask=mask)
    
    # Load b[im1] values
    b_im1_vals = tl.load(b_ptr + im1_indices.to(tl.int32), mask=mask)
    
    # Compute result
    result = (b_vals + b_im1_vals) * 0.5
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s291_triton(a, b):
    n_elements = a.shape[0]
    
    # Create expanded im1 array
    im1_expanded = torch.zeros(n_elements, dtype=torch.int32, device=a.device)
    
    # Compute expanded im1 array (single thread)
    s291_expand_im1_kernel[(1,)](im1_expanded, n_elements)
    
    # Main computation with expanded array
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s291_kernel[grid](a, b, im1_expanded, n_elements, BLOCK_SIZE)