import triton
import triton.language as tl
import torch

@triton.jit
def s31111_kernel(a_ptr, sum_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Single thread handles the entire computation since it's a simple sum
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Sum of 8 test() calls, each summing 4 consecutive elements
    total_sum = 0.0
    
    # test(a) - sum a[0:4]
    offsets = tl.arange(0, 4)
    mask = offsets < N
    vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    total_sum += tl.sum(vals)
    
    # test(&a[4]) - sum a[4:8]
    offsets = 4 + tl.arange(0, 4)
    mask = offsets < N
    vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    total_sum += tl.sum(vals)
    
    # test(&a[8]) - sum a[8:12]
    offsets = 8 + tl.arange(0, 4)
    mask = offsets < N
    vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    total_sum += tl.sum(vals)
    
    # test(&a[12]) - sum a[12:16]
    offsets = 12 + tl.arange(0, 4)
    mask = offsets < N
    vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    total_sum += tl.sum(vals)
    
    # test(&a[16]) - sum a[16:20]
    offsets = 16 + tl.arange(0, 4)
    mask = offsets < N
    vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    total_sum += tl.sum(vals)
    
    # test(&a[20]) - sum a[20:24]
    offsets = 20 + tl.arange(0, 4)
    mask = offsets < N
    vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    total_sum += tl.sum(vals)
    
    # test(&a[24]) - sum a[24:28]
    offsets = 24 + tl.arange(0, 4)
    mask = offsets < N
    vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    total_sum += tl.sum(vals)
    
    # test(&a[28]) - sum a[28:32]
    offsets = 28 + tl.arange(0, 4)
    mask = offsets < N
    vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    total_sum += tl.sum(vals)
    
    # Store the result
    tl.store(sum_ptr, total_sum)

def s31111_triton(a):
    N = a.shape[0]
    
    # Output tensor for the sum
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single thread since it's a simple reduction
    BLOCK_SIZE = 32
    grid = (1,)
    
    s31111_kernel[grid](
        a, result, N, BLOCK_SIZE
    )
    
    return result[0].item()