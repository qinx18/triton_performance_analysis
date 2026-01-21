import triton
import triton.language as tl
import torch

@triton.jit
def s316_kernel(a_ptr, result_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize result with first element
    if tl.program_id(0) == 0:
        first_val = tl.load(a_ptr)
        tl.store(result_ptr, first_val)
    
    # Process elements in blocks
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        # Load values with infinity as other value for min reduction
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('inf'))
        
        # Find block minimum
        block_min = tl.min(vals, axis=0)
        
        # Atomically update global minimum
        tl.atomic_min(result_ptr, block_min)

def s316_triton(a):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Initialize result with first element
    result = torch.tensor([a[0].item()], dtype=a.dtype, device=a.device)
    
    # Launch kernel with single program
    grid = (1,)
    s316_kernel[grid](
        a, result, n, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()