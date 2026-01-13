import torch
import triton
import triton.language as tl

@triton.jit
def expand_im1_kernel(im1_expanded_ptr, n_elements):
    # Single thread processes all elements sequentially
    im1_val = n_elements - 1  # Initial value: LEN_1D-1
    for i in range(n_elements):
        tl.store(im1_expanded_ptr + i, im1_val)
        im1_val = i  # Update for next iteration

@triton.jit
def s291_kernel(a_ptr, b_ptr, im1_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Load b values
    b_vals = tl.load(b_ptr + current_offsets, mask=mask)
    
    # Load im1 values (indices into b array)
    im1_vals = tl.load(im1_expanded_ptr + current_offsets, mask=mask)
    
    # Load b[im1] values
    b_im1_vals = tl.load(b_ptr + im1_vals.to(tl.int32), mask=mask)
    
    # Compute a[i] = (b[i] + b[im1]) * 0.5
    result = (b_vals + b_im1_vals) * 0.5
    
    # Store result
    tl.store(a_ptr + current_offsets, result, mask=mask)

def s291_triton(a, b):
    n_elements = a.shape[0]
    
    # Create expanded im1 array
    im1_expanded = torch.zeros(n_elements, dtype=torch.int32, device=a.device)
    
    # Phase 1: Expand im1 scalar to array (single thread)
    expand_im1_kernel[(1,)](im1_expanded, n_elements)
    
    # Phase 2: Use expanded array in parallel computation
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s291_kernel[grid](a, b, im1_expanded, n_elements, BLOCK_SIZE)