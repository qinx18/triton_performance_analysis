import torch
import triton
import triton.language as tl

@triton.jit
def s1113_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    scalar_val,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s1113: vectorized addition with broadcast scalar.
    Optimized for coalesced memory access and efficient broadcasting.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load b values with masking for edge cases
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Broadcast scalar addition - scalar_val is already loaded on host
    result = scalar_val + b_vals
    
    # Store result with masking
    tl.store(output_ptr + offsets, result, mask=mask)

def s1113_triton(a, b):
    """
    Triton implementation of TSVC s1113.
    Optimized with coalesced memory access and efficient scalar broadcasting.
    """
    a = a.contiguous()
    b = b.contiguous()
    
    LEN_1D = a.size(0)
    # Extract scalar value on host to avoid device-to-host transfer in kernel
    scalar_val = a[LEN_1D // 2].item()
    
    # Allocate output tensor
    output = torch.empty_like(a)
    
    n_elements = a.numel()
    
    # Use block size optimized for memory bandwidth
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel with scalar value passed directly
    s1113_kernel[grid](
        a, b, output,
        scalar_val,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output