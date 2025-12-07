import torch
import triton
import triton.language as tl

@triton.jit
def vsumr_kernel(
    a_ptr,
    result_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize sum for this block
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Process elements in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load elements from array a
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Accumulate
        acc += a_vals
    
    # Reduce within block
    block_sum = tl.sum(acc)
    
    # Store result (only first thread writes)
    if tl.program_id(0) == 0:
        tl.atomic_add(result_ptr, block_sum)

def vsumr_triton(a):
    n_elements = a.numel()
    
    # Create result tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    vsumr_kernel[grid](
        a,
        result,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return result.item()