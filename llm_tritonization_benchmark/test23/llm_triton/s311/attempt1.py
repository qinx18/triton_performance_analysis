import triton
import triton.language as tl
import torch

@triton.jit
def s311_kernel(a_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    block_sum = 0.0
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        block_sum += tl.sum(vals, axis=0)
    
    tl.store(output_ptr, block_sum)

def s311_triton(a):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    grid = (1,)
    s311_kernel[grid](a, output, n, BLOCK_SIZE)
    
    return output.item()