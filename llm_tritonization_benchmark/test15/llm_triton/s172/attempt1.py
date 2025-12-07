import torch
import triton
import triton.language as tl

@triton.jit
def s172_kernel(a_ptr, b_ptr, n1, n3, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate how many elements this thread block will process
    block_idx = tl.program_id(0)
    
    # Each block processes one stride element
    # We need to map block_idx to the actual index in the strided pattern
    actual_idx = n1 - 1 + block_idx * n3
    
    # Check if this index is valid
    if actual_idx >= n_elements:
        return
    
    # Load and compute
    a_val = tl.load(a_ptr + actual_idx)
    b_val = tl.load(b_ptr + actual_idx)
    result = a_val + b_val
    tl.store(a_ptr + actual_idx, result)

def s172_triton(a, b, n1, n3):
    n_elements = a.shape[0]
    
    # Calculate number of valid indices in the strided pattern
    start_idx = n1 - 1
    if start_idx >= n_elements or n3 <= 0:
        return
    
    # Calculate how many elements will be processed
    num_blocks = (n_elements - start_idx + n3 - 1) // n3
    
    if num_blocks <= 0:
        return
    
    BLOCK_SIZE = 1  # Each block processes one element
    
    s172_kernel[(num_blocks,)](
        a, b, n1, n3, n_elements, BLOCK_SIZE
    )