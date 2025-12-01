import triton
import triton.language as tl
import torch

@triton.jit
def s314_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Initialize with first element
    first_val = tl.load(a_ptr)
    max_val = first_val
    
    # Define offsets once at kernel start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process array in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load block of values
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=-float('inf'))
        
        # Find max in current block
        block_max = tl.max(vals, axis=0)
        
        # Update global max
        max_val = tl.maximum(max_val, block_max)
    
    # Store result
    tl.store(output_ptr, max_val)

def s314_triton(a):
    n_elements = a.shape[0]
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 1024
    
    s314_kernel[(1,)](
        a, output, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output.item()