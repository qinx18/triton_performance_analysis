import torch
import triton
import triton.language as tl

@triton.jit
def s291_expand_im1_kernel(im1_expanded_ptr, n_elements):
    # Single thread processes all elements sequentially to expand im1
    if tl.program_id(0) != 0:
        return
    
    im1_val = n_elements - 1  # Initial value: LEN_1D-1
    for i in range(n_elements):
        tl.store(im1_expanded_ptr + i, im1_val)
        im1_val = i  # im1 = i for next iteration

@triton.jit
def s291_kernel(a_ptr, b_ptr, im1_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load current b values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Load im1 values (previous indices)
    im1_vals = tl.load(im1_expanded_ptr + offsets, mask=mask)
    
    # Load b[im1] values
    b_im1_vals = tl.load(b_ptr + im1_vals, mask=mask)
    
    # Compute a[i] = (b[i] + b[im1]) * 0.5
    result = (b_vals + b_im1_vals) * 0.5
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s291_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create expanded im1 array
    im1_expanded = torch.zeros(n_elements, dtype=torch.int32, device=a.device)
    
    # Phase 1: Expand im1 scalar to array
    grid = (1,)
    s291_expand_im1_kernel[grid](im1_expanded, n_elements)
    
    # Phase 2: Parallel computation using expanded array
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s291_kernel[grid](a, b, im1_expanded, n_elements, BLOCK_SIZE)