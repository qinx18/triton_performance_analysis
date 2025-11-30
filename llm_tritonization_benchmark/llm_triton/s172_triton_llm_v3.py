import triton
import triton.language as tl
import torch

@triton.jit
def s172_kernel(a_ptr, b_ptr, n1, n3, LEN_1D, BLOCK_SIZE: tl.constexpr):
    # Calculate the starting index for this block
    block_id = tl.program_id(0)
    
    # Calculate how many elements this thread block will process
    # We need to map block_id to the actual indices in the strided pattern
    base_idx = n1 - 1 + block_id * n3
    
    # Check if this block's starting index is valid
    if base_idx >= LEN_1D:
        return
    
    # Load and compute
    mask = base_idx < LEN_1D
    if mask:
        a_val = tl.load(a_ptr + base_idx)
        b_val = tl.load(b_ptr + base_idx)
        result = a_val + b_val
        tl.store(a_ptr + base_idx, result)

def s172_triton(a, b, n1, n3):
    LEN_1D = a.shape[0]
    
    # Calculate the number of elements that will be processed
    if n3 <= 0 or n1 - 1 >= LEN_1D:
        return
    
    # Calculate number of iterations in the original loop
    num_elements = (LEN_1D - (n1 - 1) + n3 - 1) // n3
    
    BLOCK_SIZE = 1
    grid = (num_elements,)
    
    s172_kernel[grid](
        a, b, n1, n3, LEN_1D, BLOCK_SIZE
    )