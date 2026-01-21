import triton
import triton.language as tl
import torch

@triton.jit
def s172_kernel(a_ptr, b_ptr, n1, n3, N, BLOCK_SIZE: tl.constexpr):
    # Calculate how many elements this kernel will process
    num_elements = (N - (n1 - 1) + n3 - 1) // n3
    
    # Get block ID and calculate starting position
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid elements in this block
    mask = offsets < num_elements
    
    # Convert linear indices to actual array indices using the stride pattern
    # i = n1-1 + offset * n3
    actual_indices = (n1 - 1) + offsets * n3
    
    # Load values from a and b arrays
    a_vals = tl.load(a_ptr + actual_indices, mask=mask)
    b_vals = tl.load(b_ptr + actual_indices, mask=mask)
    
    # Perform the computation: a[i] += b[i]
    result = a_vals + b_vals
    
    # Store the result back to array a
    tl.store(a_ptr + actual_indices, result, mask=mask)

def s172_triton(a, b, n1, n3):
    N = a.shape[0]
    
    # Calculate the number of elements that will be processed
    num_elements = (N - (n1 - 1) + n3 - 1) // n3
    
    if num_elements <= 0:
        return
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
    
    s172_kernel[grid](a, b, n1, n3, N, BLOCK_SIZE=BLOCK_SIZE)