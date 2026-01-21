import torch
import triton
import triton.language as tl

@triton.jit
def s31111_kernel(a_ptr, result_ptr, BLOCK_SIZE: tl.constexpr):
    # Initialize sum
    sum_val = 0.0
    
    # Create offsets for loading 4 elements at a time
    offsets = tl.arange(0, 4)
    
    # test(a) - sum elements a[0] to a[3]
    vals = tl.load(a_ptr + offsets)
    sum_val += tl.sum(vals)
    
    # test(&a[4]) - sum elements a[4] to a[7]
    vals = tl.load(a_ptr + offsets + 4)
    sum_val += tl.sum(vals)
    
    # test(&a[8]) - sum elements a[8] to a[11]
    vals = tl.load(a_ptr + offsets + 8)
    sum_val += tl.sum(vals)
    
    # test(&a[12]) - sum elements a[12] to a[15]
    vals = tl.load(a_ptr + offsets + 12)
    sum_val += tl.sum(vals)
    
    # test(&a[16]) - sum elements a[16] to a[19]
    vals = tl.load(a_ptr + offsets + 16)
    sum_val += tl.sum(vals)
    
    # test(&a[20]) - sum elements a[20] to a[23]
    vals = tl.load(a_ptr + offsets + 20)
    sum_val += tl.sum(vals)
    
    # test(&a[24]) - sum elements a[24] to a[27]
    vals = tl.load(a_ptr + offsets + 24)
    sum_val += tl.sum(vals)
    
    # test(&a[28]) - sum elements a[28] to a[31]
    vals = tl.load(a_ptr + offsets + 28)
    sum_val += tl.sum(vals)
    
    # Store result
    tl.store(result_ptr, sum_val)

def s31111_triton(a):
    # Create output tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    BLOCK_SIZE = 32
    grid = (1,)
    
    s31111_kernel[grid](
        a, result,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result[0].item()