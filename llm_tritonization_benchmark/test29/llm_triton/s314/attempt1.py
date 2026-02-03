import triton
import triton.language as tl
import torch

@triton.jit
def s314_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Find maximum value in array a
    max_val = float('-inf')
    
    # Process array in blocks
    offsets = tl.arange(0, BLOCK_SIZE)
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
        block_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, block_max)
    
    # Store result
    if tl.program_id(0) == 0:
        tl.store(result_ptr, max_val)

def s314_triton(a):
    n_elements = a.shape[0]
    
    # Output tensor for result
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single program
    BLOCK_SIZE = 1024
    grid = (1,)
    
    s314_kernel[grid](
        a, result, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()