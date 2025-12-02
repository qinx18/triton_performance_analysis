import torch
import triton
import triton.language as tl

@triton.jit
def s292_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load b array for current block
    b_offsets = block_start + offsets
    b_vals = tl.load(b_ptr + b_offsets, mask=mask, other=0.0)
    
    # For each element in this block, we need b[im1] and b[im2]
    # Since im1 and im2 change sequentially, we need to process element by element
    for local_i in range(BLOCK_SIZE):
        global_i = block_start + local_i
        if global_i >= n_elements:
            break
            
        # Calculate im1 and im2 for this iteration
        if global_i == 0:
            im1 = n_elements - 1
            im2 = n_elements - 2
        elif global_i == 1:
            im1 = 0
            im2 = n_elements - 1
        else:
            im1 = global_i - 1
            im2 = global_i - 2
        
        # Load required values
        b_i = tl.load(b_ptr + global_i)
        b_im1 = tl.load(b_ptr + im1)
        b_im2 = tl.load(b_ptr + im2)
        
        # Compute result
        result = (b_i + b_im1 + b_im2) * 0.333
        
        # Store result
        tl.store(a_ptr + global_i, result)

def s292_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s292_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )