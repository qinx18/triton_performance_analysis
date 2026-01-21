import torch
import triton
import triton.language as tl

@triton.jit
def s291_expand_im1_kernel(im1_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid == 0:  # Only first thread block executes this
        im1_val = n_elements - 1  # Initial value: LEN_1D-1
        
        # Process all elements sequentially to maintain dependency
        for i in range(n_elements):
            tl.store(im1_expanded_ptr + i, im1_val)
            im1_val = i  # im1 = i after store

@triton.jit
def s291_kernel(a_ptr, b_ptr, im1_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    im1_vals = tl.load(im1_expanded_ptr + offsets, mask=mask)
    
    # Load b[im1] values - need to gather from different indices
    b_im1_vals = tl.load(b_ptr + im1_vals.to(tl.int32), mask=mask)
    
    # Compute: a[i] = (b[i] + b[im1]) * 0.5
    result = (b_vals + b_im1_vals) * 0.5
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s291_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create temporary array for expanded im1 values
    im1_expanded = torch.zeros(n_elements, dtype=torch.int32, device=a.device)
    
    # Phase 1: Expand im1 scalar to array
    grid = (1,)  # Only one thread block needed
    s291_expand_im1_kernel[grid](im1_expanded, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Phase 2: Parallel computation using expanded array
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s291_kernel[grid](a, b, im1_expanded, n_elements, BLOCK_SIZE=BLOCK_SIZE)