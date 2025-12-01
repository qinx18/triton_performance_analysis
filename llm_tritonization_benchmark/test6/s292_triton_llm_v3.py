import torch
import triton
import triton.language as tl

@triton.jit
def s292_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Load entire b array for random access
    b_data = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        block_data = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        b_data = tl.where(current_offsets < BLOCK_SIZE, block_data, b_data)
    
    # Sequential computation with wrap-around
    im1 = n_elements - 1
    im2 = n_elements - 2
    
    for i in range(n_elements):
        # Load b[i], b[im1], b[im2]
        b_i = tl.load(b_ptr + i)
        b_im1 = tl.load(b_ptr + im1)
        b_im2 = tl.load(b_ptr + im2)
        
        # Compute and store result
        result = (b_i + b_im1 + b_im2) * 0.333
        tl.store(a_ptr + i, result)
        
        # Update indices
        im2 = im1
        im1 = i

def s292_triton(a, b):
    n_elements = b.shape[0]
    BLOCK_SIZE = 256
    
    grid = (1,)
    s292_kernel[grid](
        a, b,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )