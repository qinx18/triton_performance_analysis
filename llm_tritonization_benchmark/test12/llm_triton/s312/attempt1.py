import torch
import triton
import triton.language as tl

@triton.jit
def s312_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Product reduction using a single thread block
    pid = tl.program_id(0)
    if pid != 0:  # Only use first program
        return
    
    offsets = tl.arange(0, BLOCK_SIZE)
    prod = 1.0
    
    # Process array in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=1.0)
        
        # Multiply all values in block
        for i in range(BLOCK_SIZE):
            if block_start + i < n_elements:
                val = tl.load(a_ptr + (block_start + i))
                prod *= val
    
    # Store result
    tl.store(result_ptr, prod)

def s312_triton(a):
    n_elements = a.shape[0]
    result = torch.ones(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 1024
    grid = (1,)
    
    s312_kernel[grid](
        a, result, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()