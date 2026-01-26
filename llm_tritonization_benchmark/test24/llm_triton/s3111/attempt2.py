import triton
import triton.language as tl
import torch

@triton.jit
def s3111_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    block_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        positive_mask = vals > 0.0
        block_sum = tl.where(positive_mask & mask, vals, 0.0)
        
        if block_start == 0:
            total_sum = tl.sum(block_sum)
        else:
            total_sum = total_sum + tl.sum(block_sum)
    
    tl.store(output_ptr, total_sum)

def s3111_triton(a):
    n_elements = a.shape[0]
    
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 1024
    
    s3111_kernel[(1,)](
        a, output, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output.item()