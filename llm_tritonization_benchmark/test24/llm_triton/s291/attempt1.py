import torch
import triton
import triton.language as tl

@triton.jit
def s291_expand_im1_kernel(im1_expanded_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Only first thread processes sequentially
    if pid != 0:
        return
    
    im1_val = n - 1  # Initial value: LEN_1D-1
    
    # Process all elements sequentially
    for i in range(n):
        tl.store(im1_expanded_ptr + i, im1_val)
        im1_val = i

@triton.jit
def s291_kernel(a_ptr, b_ptr, im1_expanded_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # Load b values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Load expanded im1 values
    im1_vals = tl.load(im1_expanded_ptr + offsets, mask=mask)
    
    # Load b[im1] values using im1_vals as indices
    b_im1_vals = tl.load(b_ptr + im1_vals.to(tl.int32), mask=mask)
    
    # Compute: a[i] = (b[i] + b[im1]) * 0.5
    result = (b_vals + b_im1_vals) * 0.5
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s291_triton(a, b):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create expanded im1 array
    im1_expanded = torch.zeros(n, dtype=torch.int32, device=a.device)
    
    # Phase 1: Expand scalar im1
    grid = (1,)  # Single thread
    s291_expand_im1_kernel[grid](im1_expanded, n, BLOCK_SIZE=BLOCK_SIZE)
    
    # Phase 2: Parallel computation
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    s291_kernel[grid](a, b, im1_expanded, n, BLOCK_SIZE=BLOCK_SIZE)
    
    return a