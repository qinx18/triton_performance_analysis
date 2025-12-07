import torch
import triton
import triton.language as tl

@triton.jit
def s3113_kernel(a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    block_max = float('-inf')
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        abs_vals = tl.abs(vals)
        block_max_val = tl.max(abs_vals)
        
        block_max = tl.maximum(block_max, block_max_val)
    
    tl.store(tl.program_id(0) + tl.zeros([1], dtype=tl.int32), block_max)

def s3113_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = min(1024, triton.next_power_of_2(n_elements))
    
    # Create output tensor for the result
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single block
    grid = (1,)
    s3113_kernel[grid](
        a, n_elements, BLOCK_SIZE
    )
    
    return output.item()