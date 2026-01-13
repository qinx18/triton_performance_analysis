import torch
import triton
import triton.language as tl

@triton.jit
def s291_expand_im1_kernel(im1_expanded_ptr, n):
    # Single thread processes all elements sequentially to expand im1
    im1_val = n - 1  # Initial value: LEN_1D-1
    for i in range(n):
        tl.store(im1_expanded_ptr + i, im1_val)
        im1_val = i

@triton.jit
def s291_kernel(a_ptr, b_ptr, im1_expanded_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n
    
    b_vals = tl.load(b_ptr + indices, mask=mask)
    im1_vals = tl.load(im1_expanded_ptr + indices, mask=mask)
    b_im1_vals = tl.load(b_ptr + im1_vals.to(tl.int32), mask=mask)
    
    result = (b_vals + b_im1_vals) * 0.5
    
    tl.store(a_ptr + indices, result, mask=mask)

def s291_triton(a, b):
    n = a.shape[0]
    
    # Create expanded im1 array
    im1_expanded = torch.zeros(n, dtype=torch.int32, device=a.device)
    
    # Expand im1 values
    s291_expand_im1_kernel[(1,)](
        im1_expanded,
        n
    )
    
    # Main computation
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s291_kernel[grid](
        a, b, im1_expanded, n, BLOCK_SIZE
    )