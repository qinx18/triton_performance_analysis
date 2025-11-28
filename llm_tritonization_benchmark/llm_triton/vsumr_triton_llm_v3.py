import triton
import triton.language as tl
import torch

@triton.jit
def vsumr_kernel(
    a_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values from array a
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Sum the values in this block
    block_sum = tl.sum(a_vals)
    
    # Store the partial sum (each block contributes one sum)
    if tl.program_id(0) == 0:
        tl.atomic_add(output_ptr, block_sum)

def vsumr_triton(a):
    n_elements = a.shape[0]
    
    # Create output tensor for the sum
    output = torch.zeros(1, device=a.device, dtype=a.dtype)
    
    # Choose block size
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    vsumr_kernel[grid](
        a,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output.item()