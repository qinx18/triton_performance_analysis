import triton
import triton.language as tl
import torch

@triton.jit
def s341_kernel(b_ptr, a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel packs positive values from b into a
    # Since this is inherently sequential (pack operation), we use a single thread
    pid = tl.program_id(0)
    
    if pid != 0:
        return
    
    j = -1
    for i in range(n_elements):
        b_val = tl.load(b_ptr + i)
        if b_val > 0.0:
            j += 1
            tl.store(a_ptr + j, b_val)

def s341_triton(a, b):
    n_elements = b.shape[0]
    
    # Reset output array
    a.zero_()
    
    # Use single block since this is inherently sequential
    grid = (1,)
    BLOCK_SIZE = 256
    
    s341_kernel[grid](
        b, a, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )