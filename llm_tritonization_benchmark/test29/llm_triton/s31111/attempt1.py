import torch
import triton
import triton.language as tl

@triton.jit
def s31111_kernel(a_ptr, sum_ptr, N, BLOCK_SIZE: tl.constexpr):
    # This kernel computes a single sum value by summing specific ranges
    # We'll use a single thread block to compute the entire sum
    
    pid = tl.program_id(0)
    if pid > 0:
        return
    
    # Initialize sum
    total_sum = 0.0
    
    # Sum test(a) - elements a[0:4]
    if N >= 4:
        offsets = tl.arange(0, 4)
        mask = offsets < N
        vals = tl.load(a_ptr + offsets, mask=mask)
        total_sum += tl.sum(vals)
    
    # Sum test(&a[4]) - elements a[4:8]
    if N >= 8:
        offsets = tl.arange(0, 4)
        actual_offsets = 4 + offsets
        mask = actual_offsets < N
        vals = tl.load(a_ptr + actual_offsets, mask=mask)
        total_sum += tl.sum(vals)
    
    # Sum test(&a[8]) - elements a[8:12]
    if N >= 12:
        offsets = tl.arange(0, 4)
        actual_offsets = 8 + offsets
        mask = actual_offsets < N
        vals = tl.load(a_ptr + actual_offsets, mask=mask)
        total_sum += tl.sum(vals)
    
    # Sum test(&a[12]) - elements a[12:16]
    if N >= 16:
        offsets = tl.arange(0, 4)
        actual_offsets = 12 + offsets
        mask = actual_offsets < N
        vals = tl.load(a_ptr + actual_offsets, mask=mask)
        total_sum += tl.sum(vals)
    
    # Sum test(&a[16]) - elements a[16:20]
    if N >= 20:
        offsets = tl.arange(0, 4)
        actual_offsets = 16 + offsets
        mask = actual_offsets < N
        vals = tl.load(a_ptr + actual_offsets, mask=mask)
        total_sum += tl.sum(vals)
    
    # Sum test(&a[20]) - elements a[20:24]
    if N >= 24:
        offsets = tl.arange(0, 4)
        actual_offsets = 20 + offsets
        mask = actual_offsets < N
        vals = tl.load(a_ptr + actual_offsets, mask=mask)
        total_sum += tl.sum(vals)
    
    # Sum test(&a[24]) - elements a[24:28]
    if N >= 28:
        offsets = tl.arange(0, 4)
        actual_offsets = 24 + offsets
        mask = actual_offsets < N
        vals = tl.load(a_ptr + actual_offsets, mask=mask)
        total_sum += tl.sum(vals)
    
    # Sum test(&a[28]) - elements a[28:32]
    if N >= 32:
        offsets = tl.arange(0, 4)
        actual_offsets = 28 + offsets
        mask = actual_offsets < N
        vals = tl.load(a_ptr + actual_offsets, mask=mask)
        total_sum += tl.sum(vals)
    
    # Store the result
    tl.store(sum_ptr, total_sum)

def s31111_triton(a):
    N = a.shape[0]
    
    # Create output tensor for the sum
    sum_result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single thread block
    grid = (1,)
    BLOCK_SIZE = 32
    
    s31111_kernel[grid](
        a, sum_result, N, BLOCK_SIZE
    )
    
    return sum_result[0].item()