import triton
import triton.language as tl
import torch

@triton.jit
def test_function(ptr, offset):
    # Simple test function - returns the value at the given offset
    return tl.load(ptr + offset)

@triton.jit
def s31111_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel computes the sum reduction for specific array elements
    pid = tl.program_id(axis=0)
    
    if pid == 0:  # Only one block needed for this simple reduction
        # Load elements at specific offsets and sum them
        sum_val = 0.0
        
        # Check bounds for each access
        if 0 < n_elements:
            sum_val += tl.load(a_ptr + 0)
        if 4 < n_elements:
            sum_val += tl.load(a_ptr + 4)
        if 8 < n_elements:
            sum_val += tl.load(a_ptr + 8)
        if 12 < n_elements:
            sum_val += tl.load(a_ptr + 12)
        if 16 < n_elements:
            sum_val += tl.load(a_ptr + 16)
        if 20 < n_elements:
            sum_val += tl.load(a_ptr + 20)
        if 24 < n_elements:
            sum_val += tl.load(a_ptr + 24)
        if 28 < n_elements:
            sum_val += tl.load(a_ptr + 28)
        
        tl.store(output_ptr, sum_val)

def s31111_triton(a):
    # Ensure input is contiguous
    a = a.contiguous()
    
    n_elements = a.shape[0]
    
    # Create output tensor for the sum
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single block since we only need one sum
    BLOCK_SIZE = 32
    grid = (1,)
    
    s31111_kernel[grid](
        a, 
        output, 
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output.item()