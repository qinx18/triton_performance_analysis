import torch
import triton
import triton.language as tl

@triton.jit
def test_kernel(a_ptr, offset):
    # Simple test function - returns the value at the given offset
    return tl.load(a_ptr + offset)

@triton.jit
def s31111_kernel(a_ptr, sum_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid == 0:
        # Only one thread computes the sum
        sum_val = 0.0
        sum_val += tl.load(a_ptr + 0)    # test(a)
        sum_val += tl.load(a_ptr + 4)    # test(&a[4])
        sum_val += tl.load(a_ptr + 8)    # test(&a[8])
        sum_val += tl.load(a_ptr + 12)   # test(&a[12])
        sum_val += tl.load(a_ptr + 16)   # test(&a[16])
        sum_val += tl.load(a_ptr + 20)   # test(&a[20])
        sum_val += tl.load(a_ptr + 24)   # test(&a[24])
        sum_val += tl.load(a_ptr + 28)   # test(&a[28])
        
        tl.store(sum_ptr, sum_val)

def s31111_triton(a):
    # Create output tensor for sum
    sum_out = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single block
    grid = (1,)
    BLOCK_SIZE = 32
    
    s31111_kernel[grid](
        a, sum_out,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return sum_out