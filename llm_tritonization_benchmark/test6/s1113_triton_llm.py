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
    Triton kernel for s1113: vectorized addition with broadcast scalar
    Optimized for coalesced memory access and efficient parallelization
    """
    # Calculate global thread index
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset array for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for boundary checking
    mask = offsets < n_elements
    
    # Load b values with masking for edge cases
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Compute: scalar_val + b[i] (broadcast scalar addition)
    result = scalar_val + b_vals
    
    # Store results with masking
    tl.store(output_ptr + offsets, result, mask=mask)

def s1113_triton(a, b):
    """
    Triton implementation of TSVC s1113.
    Optimized for GPU with coalesced memory access and efficient broadcasting.
    """
    a = a.contiguous()
    b = b.contiguous()
    
    LEN_1D = a.size(0)
    n_elements = LEN_1D
    
    # Extract scalar value (same as baseline)
    scalar_val = a[LEN_1D // 2]
    
    # Create output tensor
    output = torch.empty_like(a)
    
    # Choose block size for optimal memory coalescing
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel with optimized grid configuration
    s1113_kernel[grid](
        a, b, output,
        scalar_val,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output