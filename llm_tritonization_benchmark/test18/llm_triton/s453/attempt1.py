import triton
import triton.language as tl
import torch

@triton.jit
def s453_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    s = 0.0
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Sequential computation within each block
        for i in range(BLOCK_SIZE):
            if block_start + i < n_elements:
                s += 2.0
                b_val = tl.load(b_ptr + block_start + i)
                result = s * b_val
                tl.store(a_ptr + block_start + i, result)

def s453_triton(a, b):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Use single thread block since computation is inherently sequential
    grid = (1,)
    
    s453_kernel[grid](
        a, b, N, BLOCK_SIZE
    )