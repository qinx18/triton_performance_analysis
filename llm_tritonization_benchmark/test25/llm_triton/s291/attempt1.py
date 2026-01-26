import torch
import triton
import triton.language as tl

@triton.jit
def s291_expand_im1_kernel(im1_expanded_ptr, n):
    # Single thread processes all elements sequentially
    im1_val = n - 1  # Initial value: LEN_1D-1
    tl.store(im1_expanded_ptr, im1_val)
    
    for i in range(1, n):
        im1_val = i - 1  # im1 = i from previous iteration
        tl.store(im1_expanded_ptr + i, im1_val)

@triton.jit
def s291_kernel(a_ptr, b_ptr, im1_expanded_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # Load values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    im1_vals = tl.load(im1_expanded_ptr + offsets, mask=mask)
    b_im1_vals = tl.load(b_ptr + im1_vals, mask=mask)
    
    # Compute: a[i] = (b[i] + b[im1]) * 0.5
    result = (b_vals + b_im1_vals) * 0.5
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s291_triton(a, b):
    n = a.shape[0]
    
    # Create expanded im1 array
    im1_expanded = torch.zeros(n, dtype=torch.int32, device=a.device)
    
    # Phase 1: Expand scalar im1 (single thread)
    s291_expand_im1_kernel[(1,)](im1_expanded, n)
    
    # Phase 2: Parallel computation
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    s291_kernel[grid](a, b, im1_expanded, n, BLOCK_SIZE)