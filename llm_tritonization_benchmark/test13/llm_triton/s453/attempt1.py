import triton
import triton.language as tl
import torch

@triton.jit
def s453_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel handles the induction variable recognition pattern
    # Each block processes a contiguous chunk of the array
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Handle the sequential dependency s += 2.0 for each element
    for i in range(BLOCK_SIZE):
        idx = block_start + i
        if idx < n_elements:
            # s starts at 0 and adds 2 for each iteration, so s = 2 * (i + 1)
            s = 2.0 * (idx + 1)
            
            # Load b[idx] and compute a[idx] = s * b[idx]
            b_val = tl.load(b_ptr + idx)
            result = s * b_val
            tl.store(a_ptr + idx, result)

def s453_triton(a, b):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 128
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s453_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )