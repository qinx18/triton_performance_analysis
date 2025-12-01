import triton
import triton.language as tl
import torch

@triton.jit
def s291_kernel(b_ptr, a_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Handle the first element separately (i=0, im1=n-1)
    if tl.program_id(0) == 0:
        b_0 = tl.load(b_ptr + 0)
        b_im1 = tl.load(b_ptr + (n - 1))
        a_0 = (b_0 + b_im1) * 0.5
        tl.store(a_ptr + 0, a_0)
    
    # Process remaining elements in blocks
    for block_start in range(1, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        # Load b[i] values
        b_i = tl.load(b_ptr + current_offsets, mask=mask)
        
        # Load b[im1] values (im1 = i-1 for i > 0)
        im1_offsets = current_offsets - 1
        b_im1 = tl.load(b_ptr + im1_offsets, mask=mask)
        
        # Compute a[i] = (b[i] + b[im1]) * 0.5
        a_i = (b_i + b_im1) * 0.5
        
        # Store result
        tl.store(a_ptr + current_offsets, a_i, mask=mask)

def s291_triton(a, b):
    n = b.shape[0]
    BLOCK_SIZE = 256
    
    grid = (1,)
    s291_kernel[grid](b, a, n, BLOCK_SIZE)
    
    return a