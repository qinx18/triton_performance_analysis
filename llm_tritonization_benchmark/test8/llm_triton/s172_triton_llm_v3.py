import torch
import triton
import triton.language as tl

@triton.jit
def s172_kernel(
    a_ptr,
    b_ptr,
    n1,
    n3,
    LEN_1D,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate number of elements to process
    n_elements = (LEN_1D - (n1 - 1) + n3 - 1) // n3
    
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate block start
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = tl.arange(0, BLOCK_SIZE)
    block_offsets = block_start + offsets
    
    # Create mask for valid elements
    mask = block_offsets < n_elements
    
    # Calculate actual array indices: i = n1-1 + block_offset * n3
    array_indices = (n1 - 1) + block_offsets * n3
    
    # Load values with mask
    a_vals = tl.load(a_ptr + array_indices, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + array_indices, mask=mask, other=0.0)
    
    # Compute result
    result = a_vals + b_vals
    
    # Store result with mask
    tl.store(a_ptr + array_indices, result, mask=mask)

def s172_triton(a, b, n1, n3):
    LEN_1D = a.shape[0]
    
    # Calculate number of elements to process
    n_elements = (LEN_1D - (n1 - 1) + n3 - 1) // n3
    
    if n_elements <= 0:
        return
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s172_kernel[grid](
        a,
        b,
        n1,
        n3,
        LEN_1D,
        BLOCK_SIZE=BLOCK_SIZE,
    )