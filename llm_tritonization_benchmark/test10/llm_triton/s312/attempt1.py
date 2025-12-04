import torch
import triton
import triton.language as tl

@triton.jit
def s312_kernel(a_ptr, prod_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Product reduction using block-wise approach
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
    
    # Reduce within block
    block_prod = tl.reduce(local_prod, 0, tl.math.prod)
    
    # First thread stores the result
    if tl.program_id(0) == 0:
        tl.atomic_mul(prod_ptr, block_prod)

def s312_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Initialize product result
    prod = torch.tensor(1.0, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    s312_kernel[(num_blocks,)](
        a, prod, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return prod.item()