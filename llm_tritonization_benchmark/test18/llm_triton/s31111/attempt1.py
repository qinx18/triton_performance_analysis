import torch
import triton
import triton.language as tl

@triton.jit
def s31111_kernel(a_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Calculate sum for all 8 test() calls
    # test(a) + test(&a[4]) + ... + test(&a[28])
    
    sum_val = 0.0
    
    # Each test() sums 4 consecutive elements starting at different offsets
    for start_offset in range(0, 32, 4):  # 0, 4, 8, 12, 16, 20, 24, 28
        # Sum 4 elements starting at start_offset
        for i in range(4):
            idx = start_offset + i
            if idx < N:
                val = tl.load(a_ptr + idx)
                sum_val += val
    
    # Store the result
    tl.store(output_ptr, sum_val)

def s31111_triton(a):
    N = a.shape[0]
    
    # Output tensor to store the sum
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 32
    
    # Launch kernel with single block since we're computing a single sum
    s31111_kernel[(1,)](
        a, output, N, BLOCK_SIZE
    )
    
    return output.item()