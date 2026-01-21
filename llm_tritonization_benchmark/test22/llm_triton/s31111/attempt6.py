import triton
import triton.language as tl
import torch

@triton.jit
def s31111_kernel(a_ptr, sum_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Each program handles one iteration
    pid = tl.program_id(0)
    
    if pid >= 1:  # Only need one program for this computation
        return
    
    # Initialize sum
    sum_val = 0.0
    
    # Process 8 calls to test function: test(a), test(&a[4]), ..., test(&a[28])
    for offset in range(8):
        start_idx = offset * 4
        
        # Load 4 elements starting from start_idx
        offsets = tl.arange(0, 4)
        indices = start_idx + offsets
        mask = indices < N
        
        vals = tl.load(a_ptr + indices, mask=mask, other=0.0)
        
        # Sum the 4 elements (equivalent to test function)
        test_result = tl.sum(vals)
        sum_val += test_result
    
    # Store the final sum
    tl.store(sum_ptr, sum_val)

def s31111_triton(a):
    N = a.shape[0]
    
    # Output tensor for sum
    sum_tensor = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 256
    grid = (1,)  # Only need one program
    
    s31111_kernel[grid](
        a, sum_tensor, N, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return sum_tensor.item()