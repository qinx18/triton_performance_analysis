import triton
import triton.language as tl
import torch

@triton.jit
def s3111_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    vals = tl.load(a_ptr + indices, mask=mask, other=0.0)
    positive_mask = vals > 0.0
    filtered_vals = tl.where(positive_mask & mask, vals, 0.0)
    block_sum = tl.sum(filtered_vals)
    
    tl.atomic_add(output_ptr, block_sum)

def s3111_triton(a):
    n_elements = a.shape[0]
    
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s3111_kernel[grid](
        a, output, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output.item()