import triton
import triton.language as tl
import torch

@triton.jit
def s172_kernel(a_ptr, b_ptr, n1, n3, N, BLOCK_SIZE: tl.constexpr):
    # Calculate the number of elements to process
    num_elements = (N - (n1 - 1) + n3 - 1) // n3
    
    # Get program ID and calculate block start
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = tl.arange(0, BLOCK_SIZE)
    element_ids = block_start + offsets
    
    # Calculate actual array indices: i = (n1-1) + element_id * n3
    array_indices = (n1 - 1) + element_ids * n3
    
    # Create mask for valid elements
    mask = (element_ids < num_elements) & (array_indices < N)
    
    # Load values
    a_vals = tl.load(a_ptr + array_indices, mask=mask)
    b_vals = tl.load(b_ptr + array_indices, mask=mask)
    
    # Compute a[i] += b[i]
    result = a_vals + b_vals
    
    # Store result
    tl.store(a_ptr + array_indices, result, mask=mask)

def s172_triton(a, b, n1, n3):
    N = a.shape[0]
    
    # Calculate the number of elements to process
    num_elements = (N - (n1 - 1) + n3 - 1) // n3
    
    if num_elements <= 0:
        return
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid_size = triton.cdiv(num_elements, BLOCK_SIZE)
    
    # Launch kernel
    s172_kernel[(grid_size,)](
        a, b, n1, n3, N,
        BLOCK_SIZE=BLOCK_SIZE
    )