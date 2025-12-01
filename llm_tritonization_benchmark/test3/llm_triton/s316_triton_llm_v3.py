import triton
import triton.language as tl
import torch

@triton.jit
def s316_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid > 0:
        return
    
    # Initialize with first element
    x = tl.load(a_ptr)
    
    # Process remaining elements in blocks
    for block_start in range(1, n_elements, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load block of values
        values = tl.load(a_ptr + offsets, mask=mask, other=float('inf'))
        
        # Find minimum in this block
        block_min = tl.min(values)
        
        # Update global minimum
        x = tl.minimum(x, block_min)
    
    # Store result
    tl.store(output_ptr, x)

def s316_triton(a):
    n_elements = a.shape[0]
    
    # Output tensor for result
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single thread block
    BLOCK_SIZE = 1024
    grid = (1,)
    
    s316_kernel[grid](
        a, output, n_elements, BLOCK_SIZE
    )
    
    return output.item()