import torch
import triton
import triton.language as tl

@triton.jit
def s31111_kernel(a_ptr, result_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Each program computes one iteration of the original loop
    pid = tl.program_id(0)
    
    # Initialize sum
    sum_val = 0.0
    
    # Sum the 8 groups of 4 elements each (offsets 0, 4, 8, 12, 16, 20, 24, 28)
    for base_offset in range(0, 32, 4):  # 0, 4, 8, 12, 16, 20, 24, 28
        # Load 4 elements starting at base_offset
        offsets = base_offset + tl.arange(0, 4)
        mask = offsets < n
        vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
        
        # Sum the 4 elements (equivalent to test() function)
        group_sum = tl.sum(vals)
        sum_val += group_sum
    
    # Store the result
    tl.store(result_ptr + pid, sum_val)

def s31111_triton(a):
    n = a.shape[0]
    
    # We need at least 32 elements for the computation
    if n < 32:
        # Pad with zeros if needed
        a_padded = torch.zeros(32, dtype=a.dtype, device=a.device)
        a_padded[:n] = a
        a = a_padded
        n = 32
    
    # Create output tensor for single result
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single program
    grid = (1,)
    BLOCK_SIZE = 32
    
    s31111_kernel[grid](
        a, result, n,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()