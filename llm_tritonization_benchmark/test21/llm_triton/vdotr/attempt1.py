import triton
import triton.language as tl
import torch

@triton.jit
def vdotr_kernel(a_ptr, b_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    accumulator = 0.0
    
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        block_dot = tl.sum(a_vals * b_vals)
        accumulator += block_dot
    
    tl.store(output_ptr, accumulator)

def vdotr_triton(a, b):
    N = a.shape[0]
    
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 256
    
    grid = (1,)
    vdotr_kernel[grid](a, b, output, N, BLOCK_SIZE=BLOCK_SIZE)
    
    return output.item()