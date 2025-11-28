import triton
import triton.language as tl
import torch

@triton.jit
def s172_kernel(a_ptr, b_ptr, n1, n3, LEN_1D, BLOCK_SIZE: tl.constexpr):
    # Calculate number of elements in the strided sequence
    num_elements = (LEN_1D - (n1 - 1) + n3 - 1) // n3
    
    # Get program ID and calculate element index
    pid = tl.program_id(axis=0)
    
    # Calculate the actual array index using strided pattern
    i = (n1 - 1) + pid * n3
    
    # Create mask to handle bounds
    mask = (pid < num_elements) & (i < LEN_1D)
    
    # Load values with masking
    a_val = tl.load(a_ptr + i, mask=mask)
    b_val = tl.load(b_ptr + i, mask=mask)
    
    # Perform computation
    result = a_val + b_val
    
    # Store result with masking
    tl.store(a_ptr + i, result, mask=mask)

def s172_triton(a, b, n1, n3):
    LEN_1D = a.shape[0]
    
    # Calculate number of elements in the strided sequence
    num_elements = (LEN_1D - (n1 - 1) + n3 - 1) // n3
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s172_kernel[grid](
        a, b,
        n1, n3, LEN_1D,
        BLOCK_SIZE=BLOCK_SIZE
    )