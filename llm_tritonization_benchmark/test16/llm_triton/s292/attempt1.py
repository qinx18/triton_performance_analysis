import triton
import triton.language as tl
import torch

@triton.jit
def s292_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    if pid > 0:
        return
        
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Load entire b array for random access
    b_vals = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        b_block = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        b_vals = tl.where(mask, b_block, b_vals)
    
    # Sequential computation
    im1 = n_elements - 1
    im2 = n_elements - 2
    
    for i in range(n_elements):
        # Load b[i]
        b_i = tl.load(b_ptr + i)
        
        # Load b[im1] and b[im2]
        b_im1 = tl.load(b_ptr + im1)
        b_im2 = tl.load(b_ptr + im2)
        
        # Compute result
        result = (b_i + b_im1 + b_im2) * 0.333
        
        # Store result
        tl.store(a_ptr + i, result)
        
        # Update indices
        im2 = im1
        im1 = i

def s292_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    grid = (1,)
    
    s292_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )