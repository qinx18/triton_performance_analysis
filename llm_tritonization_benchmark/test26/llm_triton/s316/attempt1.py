import triton
import triton.language as tl
import torch

@triton.jit
def s316_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Initialize offsets
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize block minimum with positive infinity
    block_min = float('inf')
    
    # Process all elements in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values with infinity as default for out-of-bounds
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('inf'))
        
        # Update block minimum
        current_min = tl.min(vals, axis=0)
        if current_min < block_min:
            block_min = current_min
    
    # Store the result
    tl.store(result_ptr, block_min)

def s316_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Create output tensor
    result = torch.empty(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single thread block since we need global minimum
    s316_kernel[(1,)](
        a, result, n_elements, BLOCK_SIZE
    )
    
    return result.item()