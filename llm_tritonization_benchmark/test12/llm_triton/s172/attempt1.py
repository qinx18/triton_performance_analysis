import torch
import triton
import triton.language as tl

@triton.jit
def s172_kernel(a_ptr, b_ptr, n1, n3, LEN_1D):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate the actual index based on stride pattern
    # Each program handles one element in the strided sequence
    i = n1 - 1 + pid * n3
    
    # Check bounds
    if i < LEN_1D:
        # Load values
        a_val = tl.load(a_ptr + i)
        b_val = tl.load(b_ptr + i)
        
        # Perform computation
        result = a_val + b_val
        
        # Store result
        tl.store(a_ptr + i, result)

def s172_triton(a, b, n1, n3):
    LEN_1D = a.shape[0]
    
    # Calculate number of elements that will be processed
    if n3 <= 0:
        return
    
    start_idx = n1 - 1
    if start_idx >= LEN_1D or start_idx < 0:
        return
    
    # Calculate how many elements will be processed
    num_elements = (LEN_1D - start_idx + n3 - 1) // n3
    
    if num_elements <= 0:
        return
    
    # Launch kernel
    grid = (num_elements,)
    s172_kernel[grid](a, b, n1, n3, LEN_1D)