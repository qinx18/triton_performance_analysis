import triton
import triton.language as tl
import torch

@triton.jit
def s315_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize with first element
    first_val = tl.load(a_ptr)
    max_val = first_val
    max_idx = 0
    
    # Process array in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
        
        # Find local maximum in this block
        for i in range(BLOCK_SIZE):
            if block_start + i < n_elements:
                current_idx = block_start + i
                current_val = tl.load(a_ptr + current_idx)
                
                if current_val > max_val:
                    max_val = current_val
                    max_idx = current_idx
    
    # Store result
    result = max_val + max_idx + 1
    tl.store(output_ptr, result)

def s315_triton(a):
    n_elements = a.shape[0]
    
    # Allocate output tensor
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 1024
    grid = (1,)
    
    s315_kernel[grid](
        a, output, n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output