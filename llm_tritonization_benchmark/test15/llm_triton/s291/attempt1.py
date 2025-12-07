import torch
import triton
import triton.language as tl

@triton.jit
def s291_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Process elements in this block
    for i in range(BLOCK_SIZE):
        idx = block_start + i
        if idx >= n_elements:
            break
            
        # Determine im1 value
        if idx == 0:
            im1 = n_elements - 1
        else:
            im1 = idx - 1
            
        # Load b[idx] and b[im1]
        b_i = tl.load(b_ptr + idx)
        b_im1 = tl.load(b_ptr + im1)
        
        # Compute and store result
        result = (b_i + b_im1) * 0.5
        tl.store(a_ptr + idx, result)

def s291_triton(a, b):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s291_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )