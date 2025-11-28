import torch
import triton
import triton.language as tl

@triton.jit
def s311_kernel(
    a_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate the block start
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load values from array a
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Sum reduction within the block
    block_sum = tl.sum(a_vals)
    
    # Store the partial sum (only the first thread in the block)
    if tl.program_id(axis=0) == pid:
        tl.atomic_add(output_ptr, block_sum)

def s311_triton(a):
    n_elements = a.shape[0]
    
    # Initialize output tensor
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch parameters
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    s311_kernel[grid](
        a,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output.item()