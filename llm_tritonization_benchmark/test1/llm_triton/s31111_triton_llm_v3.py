import triton
import triton.language as tl
import torch

def test(x):
    return x

@triton.jit
def s31111_kernel(a_ptr, sum_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel processes the sum reduction pattern from the original code
    # Each block will handle one iteration of the sum computation
    
    pid = tl.program_id(0)
    
    if pid == 0:  # Only one block needed for this simple reduction
        # Load the 8 specific elements: a[0], a[4], a[8], a[12], a[16], a[20], a[24], a[28]
        offsets = tl.arange(0, 8) * 4  # [0, 4, 8, 12, 16, 20, 24, 28]
        mask = offsets < n_elements
        
        # Load values from array a at the specific offsets
        values = tl.load(a_ptr + offsets, mask=mask, other=0.0)
        
        # Sum all the loaded values
        sum_val = tl.sum(values)
        
        # Store the result
        tl.store(sum_ptr, sum_val)

def s31111_triton(a):
    # Ensure input is contiguous
    a = a.contiguous()
    
    n_elements = a.numel()
    
    # Output tensor for the sum
    sum_result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single block since we're doing a simple reduction
    BLOCK_SIZE = 32
    grid = (1,)
    
    s31111_kernel[grid](
        a, sum_result, n_elements, BLOCK_SIZE
    )
    
    return sum_result.item()