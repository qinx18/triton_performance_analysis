import triton
import triton.language as tl
import torch

@triton.jit
def s172_kernel(a_ptr, b_ptr, n1, n3, N, BLOCK_SIZE: tl.constexpr):
    # Calculate how many elements we'll process in this stride pattern
    num_elements = (N - (n1 - 1) + n3 - 1) // n3
    
    # Block-level processing
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Calculate actual array indices using the stride pattern
    element_indices = block_start + offsets
    array_indices = (n1 - 1) + element_indices * n3
    
    # Mask for valid elements
    mask = (element_indices < num_elements) & (array_indices < N)
    
    # Load values
    a_vals = tl.load(a_ptr + array_indices, mask=mask)
    b_vals = tl.load(b_ptr + array_indices, mask=mask)
    
    # Compute and store
    result = a_vals + b_vals
    tl.store(a_ptr + array_indices, result, mask=mask)

def s172_triton(a, b, n1, n3):
    N = a.shape[0]
    
    # Calculate number of elements in stride pattern
    if n1 - 1 >= N or n3 <= 0:
        return
    
    num_elements = (N - (n1 - 1) + n3 - 1) // n3
    
    if num_elements <= 0:
        return
    
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(num_elements, BLOCK_SIZE)
    
    s172_kernel[(num_blocks,)](
        a, b, n1, n3, N, BLOCK_SIZE
    )