import triton
import triton.language as tl
import torch

@triton.jit
def s341_kernel(
    a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    # This is a packing operation that cannot be parallelized due to data dependencies
    # Each positive value from b needs to be packed consecutively into a
    # We need to process sequentially to maintain the packing order
    
    offsets = tl.arange(0, BLOCK_SIZE)
    j = 0
    
    # Process in blocks but maintain sequential order
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load block of b values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process each element in the block sequentially
        for i in range(BLOCK_SIZE):
            if block_start + i < n_elements:
                b_val = tl.load(b_ptr + block_start + i)
                if b_val > 0.0:
                    tl.store(a_ptr + j, b_val)
                    j += 1

def s341_triton(a, b):
    n_elements = b.shape[0]
    BLOCK_SIZE = 1024
    
    # Initialize output array
    a.zero_()
    
    # Launch kernel with single program instance since this is sequential
    grid = (1,)
    s341_kernel[grid](
        a, b, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a