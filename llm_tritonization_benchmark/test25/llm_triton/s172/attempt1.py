import triton
import triton.language as tl
import torch

@triton.jit
def s172_kernel(a_ptr, b_ptr, n1, n3, N, BLOCK_SIZE: tl.constexpr):
    # Calculate total number of elements to process
    num_elements = (N - n1 + n3) // n3
    
    # Get program ID and calculate block boundaries
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    block_end = block_start + BLOCK_SIZE
    
    # Create offset vector for this block
    offsets = tl.arange(0, BLOCK_SIZE)
    element_indices = block_start + offsets
    
    # Mask for valid elements
    mask = element_indices < num_elements
    
    # Convert element indices to actual array indices
    array_indices = (n1 - 1) + element_indices * n3
    
    # Load values from both arrays
    a_vals = tl.load(a_ptr + array_indices, mask=mask)
    b_vals = tl.load(b_ptr + array_indices, mask=mask)
    
    # Perform computation
    result = a_vals + b_vals
    
    # Store result back to array a
    tl.store(a_ptr + array_indices, result, mask=mask)

def s172_triton(a, b, n1, n3):
    N = a.shape[0]
    
    # Calculate number of elements to process
    num_elements = (N - n1 + n3) // n3
    
    if num_elements <= 0:
        return
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid_size = triton.cdiv(num_elements, BLOCK_SIZE)
    
    # Launch kernel
    s172_kernel[(grid_size,)](
        a, b, n1, n3, N, BLOCK_SIZE
    )