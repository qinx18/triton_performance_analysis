import triton
import triton.language as tl
import torch

@triton.jit
def s172_kernel(a_ptr, b_ptr, n1, n3, N, BLOCK_SIZE: tl.constexpr):
    # Calculate the number of elements in the strided loop
    num_elements = (N - (n1 - 1) + n3 - 1) // n3
    
    # Get program ID and calculate block start
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = tl.arange(0, BLOCK_SIZE)
    element_idx = block_start + offsets
    
    # Mask for valid elements
    mask = element_idx < num_elements
    
    # Calculate actual array indices (strided pattern)
    array_indices = (n1 - 1) + element_idx * n3
    
    # Additional mask to ensure we don't go beyond array bounds
    array_mask = mask & (array_indices < N)
    
    # Load values
    a_vals = tl.load(a_ptr + array_indices, mask=array_mask)
    b_vals = tl.load(b_ptr + array_indices, mask=array_mask)
    
    # Compute
    result = a_vals + b_vals
    
    # Store result
    tl.store(a_ptr + array_indices, result, mask=array_mask)

def s172_triton(a, b, n1, n3):
    N = a.shape[0]
    
    # Calculate the number of elements that will be processed
    if n3 <= 0:
        return  # Not vectorizable if n3 == 0
    
    num_elements = (N - (n1 - 1) + n3 - 1) // n3
    
    if num_elements <= 0:
        return
    
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(num_elements, BLOCK_SIZE)
    
    s172_kernel[(num_blocks,)](
        a, b, n1, n3, N, BLOCK_SIZE
    )