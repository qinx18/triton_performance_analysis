import triton
import triton.language as tl
import torch

@triton.jit
def s31111_kernel(a_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Calculate sum of test() calls at different offsets
    sum_val = 0.0
    
    # test(a) - sum first 4 elements starting at index 0
    for i in range(4):
        if i < N:
            val = tl.load(a_ptr + i)
            sum_val += val
    
    # test(&a[4]) - sum 4 elements starting at index 4
    for i in range(4):
        idx = 4 + i
        if idx < N:
            val = tl.load(a_ptr + idx)
            sum_val += val
    
    # test(&a[8]) - sum 4 elements starting at index 8
    for i in range(4):
        idx = 8 + i
        if idx < N:
            val = tl.load(a_ptr + idx)
            sum_val += val
    
    # test(&a[12]) - sum 4 elements starting at index 12
    for i in range(4):
        idx = 12 + i
        if idx < N:
            val = tl.load(a_ptr + idx)
            sum_val += val
    
    # test(&a[16]) - sum 4 elements starting at index 16
    for i in range(4):
        idx = 16 + i
        if idx < N:
            val = tl.load(a_ptr + idx)
            sum_val += val
    
    # test(&a[20]) - sum 4 elements starting at index 20
    for i in range(4):
        idx = 20 + i
        if idx < N:
            val = tl.load(a_ptr + idx)
            sum_val += val
    
    # test(&a[24]) - sum 4 elements starting at index 24
    for i in range(4):
        idx = 24 + i
        if idx < N:
            val = tl.load(a_ptr + idx)
            sum_val += val
    
    # test(&a[28]) - sum 4 elements starting at index 28
    for i in range(4):
        idx = 28 + i
        if idx < N:
            val = tl.load(a_ptr + idx)
            sum_val += val
    
    # Only thread 0 writes the final sum
    if tl.program_id(0) == 0:
        tl.store(output_ptr, sum_val)

def s31111_triton(a):
    N = a.shape[0]
    
    # Create output tensor for the sum
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single thread since we're just computing one sum
    BLOCK_SIZE = 32
    grid = (1,)
    
    s31111_kernel[grid](
        a, output, N, BLOCK_SIZE
    )
    
    return output.item()