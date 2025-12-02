import torch
import triton
import triton.language as tl

@triton.jit
def s291_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Handle first iteration (i=0) separately
    if tl.program_id(0) == 0:
        mask_first = offsets == 0
        b_0 = tl.load(b_ptr, mask=mask_first)
        b_last = tl.load(b_ptr + (n_elements - 1), mask=mask_first)
        a_0 = (b_0 + b_last) * 0.5
        tl.store(a_ptr, a_0, mask=mask_first)
    
    # Handle remaining iterations sequentially
    for i in range(1, n_elements):
        block_id = tl.program_id(0)
        if block_id == 0 and offsets == i:
            mask = offsets == i
            b_i = tl.load(b_ptr + i, mask=mask)
            b_im1 = tl.load(b_ptr + (i - 1), mask=mask)
            a_i = (b_i + b_im1) * 0.5
            tl.store(a_ptr + i, a_i, mask=mask)

def s291_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 128
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s291_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )