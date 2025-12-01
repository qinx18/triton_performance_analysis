import torch
import triton
import triton.language as tl

@triton.jit
def vtv_kernel(
    a_ptr,
    b_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for element-wise multiplication: a[i] *= b[i]
    Uses coalesced memory access with vectorized loads/stores
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for handling array bounds
    mask = offsets < n_elements
    
    # Vectorized loads with masking
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Element-wise multiplication
    result = a_vals * b_vals
    
    # Vectorized store with masking
    tl.store(a_ptr + offsets, result, mask=mask)

def vtv_triton(a, b):
    """
    Triton implementation of TSVC vtv function.
    Performs in-place element-wise multiplication: a *= b
    """
    # Ensure contiguous memory layout for optimal memory access
    a = a.contiguous()
    b = b.contiguous()
    
    n_elements = a.numel()
    
    # Optimal block size for memory bandwidth utilization
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel with flattened tensors for linear addressing
    vtv_kernel[grid](
        a.view(-1),
        b.view(-1), 
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a