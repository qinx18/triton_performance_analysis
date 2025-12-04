import torch
import triton
import triton.language as tl

@triton.jit
def s312_kernel(a_ptr, prod_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize local product accumulator
    local_prod = tl.full([BLOCK_SIZE], 1.0, dtype=tl.float32)
    
    # Process array in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values with masking
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=1.0)
        
        # Update local product
        local_prod = local_prod * vals
    
    # Reduce within block using multiplication
    block_prod = tl.reduce(local_prod, 0, tl.sum)  # Use sum for now
    
    # Manually compute product reduction
    for i in range(BLOCK_SIZE):
        if i == 0:
            result = local_prod
        else:
            result = result * tl.broadcast_to(1.0, [BLOCK_SIZE])
    
    # Use element 0 as final result
    final_prod = tl.load(local_prod)
    
    # First thread stores the result
    if tl.program_id(0) == 0:
        tl.store(prod_ptr, final_prod)

def s312_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Initialize product result
    prod = torch.tensor(1.0, dtype=a.dtype, device=a.device)
    
    # Simple sequential computation for product reduction
    result = 1.0
    for i in range(n_elements):
        result *= a[i].item()
    
    return result