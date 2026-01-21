import triton
import triton.language as tl
import torch

@triton.jit
def s31111_kernel(a_ptr, sum_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid >= 1:
        return
    
    # Initialize sum
    sum_val = 0.0
    
    # Process 8 calls to test function
    for offset in range(8):
        start_idx = offset * 4
        
        # Ensure we don't go out of bounds
        if start_idx + 3 < N:
            # Load exactly 4 elements for test function
            offsets = tl.arange(0, 4)
            indices = start_idx + offsets
            
            vals = tl.load(a_ptr + indices)
            
            # Sum the 4 elements (test function)
            test_result = tl.sum(vals)
            sum_val += test_result
    
    # Store the final sum
    tl.store(sum_ptr, sum_val)

def s31111_triton(a):
    N = a.shape[0]
    
    # Output tensor for sum
    sum_tensor = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 256
    grid = (1,)
    
    s31111_kernel[grid](
        a, sum_tensor, N, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return sum_tensor.item()