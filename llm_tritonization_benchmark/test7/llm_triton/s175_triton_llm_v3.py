import torch
import triton
import triton.language as tl

@triton.jit
def s175_kernel(a_ptr, b_ptr, inc, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for i in range(0, n_elements, inc):
        current_idx = block_start * inc + i
        if current_idx >= n_elements - 1:
            break
            
        idx_offsets = current_idx + offsets * inc
        mask = (idx_offsets < n_elements - 1) & (offsets < 1)
        
        if tl.sum(mask.to(tl.int32)) > 0:
            a_val = tl.load(a_ptr + idx_offsets + inc, mask=mask)
            b_val = tl.load(b_ptr + idx_offsets, mask=mask)
            result = a_val + b_val
            tl.store(a_ptr + idx_offsets, result, mask=mask)

def s175_triton(a, b, inc):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    num_blocks = triton.cdiv(triton.cdiv(n_elements - 1, inc), BLOCK_SIZE)
    if num_blocks == 0:
        num_blocks = 1
    
    grid = (num_blocks,)
    
    s175_kernel[grid](a, b, inc, n_elements, BLOCK_SIZE=BLOCK_SIZE)