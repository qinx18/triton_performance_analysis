import torch
import triton
import triton.language as tl

@triton.jit
def vtvtv_kernel(
    a_ptr, b_ptr, c_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for element-wise multiplication: a[i] = a[i] * b[i] * c[i]
    """
    # Get program ID and calculate offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for boundary handling
    mask = offsets < n_elements
    
    # Load data with masking to handle edge cases
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    
    # Perform element-wise multiplication
    result = a * b * c
    
    # Store result with masking
    tl.store(output_ptr + offsets, result, mask=mask)

def vtvtv_triton(a, b, c):
    """
    Triton implementation of TSVC vtvtv function.
    Optimized for GPU with coalesced memory access and efficient blocking.
    """
    # Ensure tensors are contiguous for optimal memory access
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    
    # Get tensor properties
    n_elements = a.numel()
    
    # Create output tensor
    output = torch.empty_like(a)
    
    # Choose block size for optimal occupancy
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel with coalesced memory access pattern
    vtvtv_kernel[grid](
        a, b, c, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output