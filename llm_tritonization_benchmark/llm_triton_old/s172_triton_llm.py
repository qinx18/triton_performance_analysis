import torch
import triton
import triton.language as tl

@triton.jit
def s172_kernel(
    a_ptr,
    b_ptr,
    n_elements,
    start_idx,
    stride,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for strided vector addition.
    Updates a[i] += b[i] for i = start_idx, start_idx + stride, start_idx + 2*stride, ...
    """
    # Get program ID and calculate block start
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Generate offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Calculate actual array indices using strided pattern
    actual_indices = start_idx + offsets * stride
    
    # Create mask to handle out-of-bounds access
    mask = (offsets < n_elements) & (actual_indices >= start_idx)
    
    # Load values from both arrays with masking
    a_vals = tl.load(a_ptr + actual_indices, mask=mask)
    b_vals = tl.load(b_ptr + actual_indices, mask=mask)
    
    # Perform addition
    result = a_vals + b_vals
    
    # Store result back to array a
    tl.store(a_ptr + actual_indices, result, mask=mask)

def s172_triton(a, b, n1, n3):
    """
    Triton implementation of TSVC s172.
    Performs strided vector addition: a[i] += b[i] for i = n1-1, n1-1+n3, n1-1+2*n3, ...
    """
    a = a.contiguous()
    b = b.contiguous()
    
    start_idx = n1 - 1
    
    # Early return if start index is out of bounds
    if start_idx >= a.size(0):
        return a
    
    # Calculate number of elements that will be processed
    n_elements = (a.size(0) - start_idx + n3 - 1) // n3
    
    if n_elements <= 0:
        return a
    
    # Choose block size based on problem size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s172_kernel[grid](
        a_ptr=a,
        b_ptr=b,
        n_elements=n_elements,
        start_idx=start_idx,
        stride=n3,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a