import triton
import triton.language as tl
import torch

@triton.jit
def s172_kernel(
    a_ptr,
    b_ptr,
    n1,
    n3,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate number of elements in the strided sequence
    num_elements = (N - (n1 - 1) + n3 - 1) // n3
    
    # Get block start
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    element_ids = block_start + offsets
    
    # Mask for valid elements
    mask = element_ids < num_elements
    
    # Convert element IDs to actual array indices
    array_indices = (n1 - 1) + element_ids * n3
    
    # Load values
    a_vals = tl.load(a_ptr + array_indices, mask=mask)
    b_vals = tl.load(b_ptr + array_indices, mask=mask)
    
    # Compute
    result = a_vals + b_vals
    
    # Store result
    tl.store(a_ptr + array_indices, result, mask=mask)

def s172_triton(a, b, n1, n3):
    N = a.shape[0]
    
    # Calculate number of elements that will be processed
    if n3 <= 0:
        return  # Not vectorizable when n3 <= 0
    
    num_elements = (N - (n1 - 1) + n3 - 1) // n3
    
    if num_elements <= 0:
        return
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
    
    s172_kernel[grid](
        a, b,
        n1, n3,
        N,
        BLOCK_SIZE,
    )