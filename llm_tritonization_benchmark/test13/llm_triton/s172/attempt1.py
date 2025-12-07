import torch
import triton
import triton.language as tl

@triton.jit
def s172_kernel(a_ptr, b_ptr, n1, n3, LEN_1D, BLOCK_SIZE: tl.constexpr):
    # Calculate total number of elements to process
    start_idx = n1 - 1
    if start_idx >= LEN_1D:
        return
    
    num_elements = (LEN_1D - start_idx + n3 - 1) // n3
    
    # Get block index and calculate range
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    if block_start >= num_elements:
        return
    
    # Create offsets for this block
    offsets = tl.arange(0, BLOCK_SIZE)
    element_indices = block_start + offsets
    
    # Calculate actual array indices using stride pattern
    actual_indices = start_idx + element_indices * n3
    
    # Create mask for valid elements
    mask = (element_indices < num_elements) & (actual_indices < LEN_1D)
    
    # Load values
    a_vals = tl.load(a_ptr + actual_indices, mask=mask)
    b_vals = tl.load(b_ptr + actual_indices, mask=mask)
    
    # Compute and store
    result = a_vals + b_vals
    tl.store(a_ptr + actual_indices, result, mask=mask)

def s172_triton(a, b, n1, n3):
    LEN_1D = a.shape[0]
    
    # Early return if no valid elements
    start_idx = n1 - 1
    if start_idx >= LEN_1D or n3 <= 0:
        return
    
    # Calculate number of elements to process
    num_elements = (LEN_1D - start_idx + n3 - 1) // n3
    if num_elements <= 0:
        return
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
    
    s172_kernel[grid](
        a, b, n1, n3, LEN_1D,
        BLOCK_SIZE=BLOCK_SIZE
    )