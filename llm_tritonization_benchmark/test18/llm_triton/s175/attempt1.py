import triton
import triton.language as tl
import torch

@triton.jit
def s175_kernel(a_ptr, a_copy_ptr, b_ptr, inc, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for i in range(0, n - 1, inc):
        if i >= block_start and i < block_start + BLOCK_SIZE:
            offset = i - block_start
            if offset >= 0 and offset < BLOCK_SIZE and i < n - 1:
                mask = offset < BLOCK_SIZE
                a_copy_val = tl.load(a_copy_ptr + i, mask=mask)
                a_inc_val = tl.load(a_copy_ptr + i + inc, mask=mask and (i + inc < n))
                b_val = tl.load(b_ptr + i, mask=mask)
                result = a_inc_val + b_val
                tl.store(a_ptr + i, result, mask=mask)

def s175_triton(a, b, inc):
    n = a.shape[0]
    a_copy = a.clone()
    
    # Calculate number of elements to process (up to n-1 with stride inc)
    num_elements = ((n - 2) // inc + 1) if n > 1 else 0
    
    if num_elements <= 0:
        return
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s175_kernel[grid](
        a, a_copy, b, inc, n,
        BLOCK_SIZE=BLOCK_SIZE
    )