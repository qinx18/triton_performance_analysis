import triton
import triton.language as tl
import torch

@triton.jit
def vsumr_kernel(a_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    partial_sum = 0.0
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        partial_sum += tl.sum(vals)
    
    if tl.program_id(0) == 0:
        tl.store(output_ptr, partial_sum)

def vsumr_triton(a):
    n = a.shape[0]
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 1024
    grid = (1,)
    
    vsumr_kernel[grid](a, output, n, BLOCK_SIZE=BLOCK_SIZE)
    
    return output.item()