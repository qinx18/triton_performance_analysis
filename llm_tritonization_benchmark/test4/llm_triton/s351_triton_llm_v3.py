import triton
import triton.language as tl
import torch

@triton.jit
def s351_kernel(a_ptr, b_ptr, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for i in range(0, n_elements, 5):
        # Process 5 elements at a time
        for j in range(5):
            idx = i + j
            if idx < n_elements:
                a_val = tl.load(a_ptr + idx)
                b_val = tl.load(b_ptr + idx)
                result = a_val + alpha * b_val
                tl.store(a_ptr + idx, result)

def s351_triton(a, b, c):
    n_elements = a.shape[0]
    alpha = c[0].item()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s351_kernel[grid](
        a, b, alpha, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )