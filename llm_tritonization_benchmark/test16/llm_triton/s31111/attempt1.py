import triton
import triton.language as tl
import torch

@triton.jit
def s31111_kernel(a_ptr, sum_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid != 0:
        return
    
    # Calculate sum of elements at indices 0,1,2,3 (test(a))
    offsets_0 = tl.arange(0, 4)
    mask_0 = offsets_0 < N
    vals_0 = tl.load(a_ptr + offsets_0, mask=mask_0, other=0.0)
    sum_0 = tl.sum(vals_0)
    
    # Calculate sum of elements at indices 4,5,6,7 (test(&a[4]))
    offsets_4 = 4 + tl.arange(0, 4)
    mask_4 = offsets_4 < N
    vals_4 = tl.load(a_ptr + offsets_4, mask=mask_4, other=0.0)
    sum_4 = tl.sum(vals_4)
    
    # Calculate sum of elements at indices 8,9,10,11 (test(&a[8]))
    offsets_8 = 8 + tl.arange(0, 4)
    mask_8 = offsets_8 < N
    vals_8 = tl.load(a_ptr + offsets_8, mask=mask_8, other=0.0)
    sum_8 = tl.sum(vals_8)
    
    # Calculate sum of elements at indices 12,13,14,15 (test(&a[12]))
    offsets_12 = 12 + tl.arange(0, 4)
    mask_12 = offsets_12 < N
    vals_12 = tl.load(a_ptr + offsets_12, mask=mask_12, other=0.0)
    sum_12 = tl.sum(vals_12)
    
    # Calculate sum of elements at indices 16,17,18,19 (test(&a[16]))
    offsets_16 = 16 + tl.arange(0, 4)
    mask_16 = offsets_16 < N
    vals_16 = tl.load(a_ptr + offsets_16, mask=mask_16, other=0.0)
    sum_16 = tl.sum(vals_16)
    
    # Calculate sum of elements at indices 20,21,22,23 (test(&a[20]))
    offsets_20 = 20 + tl.arange(0, 4)
    mask_20 = offsets_20 < N
    vals_20 = tl.load(a_ptr + offsets_20, mask=mask_20, other=0.0)
    sum_20 = tl.sum(vals_20)
    
    # Calculate sum of elements at indices 24,25,26,27 (test(&a[24]))
    offsets_24 = 24 + tl.arange(0, 4)
    mask_24 = offsets_24 < N
    vals_24 = tl.load(a_ptr + offsets_24, mask=mask_24, other=0.0)
    sum_24 = tl.sum(vals_24)
    
    # Calculate sum of elements at indices 28,29,30,31 (test(&a[28]))
    offsets_28 = 28 + tl.arange(0, 4)
    mask_28 = offsets_28 < N
    vals_28 = tl.load(a_ptr + offsets_28, mask=mask_28, other=0.0)
    sum_28 = tl.sum(vals_28)
    
    # Sum all partial sums
    total_sum = sum_0 + sum_4 + sum_8 + sum_12 + sum_16 + sum_20 + sum_24 + sum_28
    
    tl.store(sum_ptr, total_sum)

def s31111_triton(a):
    N = a.shape[0]
    
    # Output tensor for the sum
    sum_result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    BLOCK_SIZE = 128
    grid = (1,)
    
    s31111_kernel[grid](
        a, sum_result, N, BLOCK_SIZE
    )
    
    return sum_result[0].item()