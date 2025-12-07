import triton
import triton.language as tl
import torch

@triton.jit
def s316_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This is a reduction to find minimum - use single thread
    pid = tl.program_id(axis=0)
    
    if pid == 0:
        # Initialize with first element
        min_val = tl.load(a_ptr)
        
        # Process remaining elements in blocks
        offsets = tl.arange(0, BLOCK_SIZE)
        
        for block_start in range(1, n_elements, BLOCK_SIZE):
            current_offsets = block_start + offsets
            mask = current_offsets < n_elements
            
            if tl.sum(mask.to(tl.int32)) > 0:
                vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('inf'))
                block_min = tl.min(vals)
                min_val = tl.minimum(min_val, block_min)
        
        tl.store(output_ptr, min_val)

def s316_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Output tensor for the minimum value
    output = torch.empty(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single thread block
    grid = (1,)
    s316_kernel[grid](
        a, output, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output.item()