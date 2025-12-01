import triton
import triton.language as tl
import torch

@triton.jit
def s292_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Sequential computation due to wrap-around dependencies
    for i in range(n_elements):
        if pid == 0:  # Only first block handles the computation
            # Compute im1 and im2 for current iteration
            if i == 0:
                im1 = n_elements - 1
                im2 = n_elements - 2
            elif i == 1:
                im1 = 0
                im2 = n_elements - 1
            else:
                im1 = i - 1
                im2 = i - 2
            
            # Load values
            b_i = tl.load(b_ptr + i)
            b_im1 = tl.load(b_ptr + im1)
            b_im2 = tl.load(b_ptr + im2)
            
            # Compute and store
            result = (b_i + b_im1 + b_im2) * 0.333
            tl.store(a_ptr + i, result)

def s292_triton(a, b):
    n_elements = a.shape[0]
    
    # Use a single block since computation must be sequential
    BLOCK_SIZE = triton.next_power_of_2(n_elements)
    grid = (1,)
    
    s292_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a