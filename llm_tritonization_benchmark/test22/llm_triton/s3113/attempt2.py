import triton
import triton.language as tl
import torch

@triton.jit
def s3113_kernel(a_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    abs_vals = tl.abs(vals)
    
    tl.store(output_ptr + offsets, abs_vals, mask=mask)

def s3113_triton(a):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create output tensor for absolute values
    abs_values = torch.empty_like(a)
    
    # Launch kernel to compute absolute values
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    s3113_kernel[grid](a, abs_values, n, BLOCK_SIZE=BLOCK_SIZE)
    
    # Find maximum using PyTorch
    max_val = torch.max(abs_values)
    
    return max_val