import torch
import triton
import triton.language as tl

@triton.jit
def vsumr_kernel(a_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    block_sum = tl.sum(vals)
    tl.atomic_add(output_ptr, block_sum)

def vsumr_triton(a):
    n = a.shape[0]
    BLOCK_SIZE = 1024
    
    output = torch.zeros(1, device=a.device, dtype=a.dtype)
    
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    vsumr_kernel[grid](a, output, n, BLOCK_SIZE=BLOCK_SIZE)
    
    return output[0]