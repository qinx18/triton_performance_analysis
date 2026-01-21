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
    
    # Load offsets for test function (4 elements)
    test_offsets = tl.arange(0, 4)
    
    # Process 8 calls to test function at indices: 0, 4, 8, 12, 16, 20, 24, 28
    for call_idx in range(8):
        start_idx = call_idx * 4
        
        # Calculate indices for this test call
        indices = start_idx + test_offsets
        
        # Check bounds - ensure we don't read beyond array bounds
        mask = indices < N
        
        # Load 4 elements with proper bounds checking
        vals = tl.load(a_ptr + indices, mask=mask, other=0.0)
        
        # Sum the 4 elements (test function implementation)
        test_result = tl.sum(vals, axis=0)
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