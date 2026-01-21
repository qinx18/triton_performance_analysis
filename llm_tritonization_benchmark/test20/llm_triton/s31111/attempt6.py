import triton
import triton.language as tl
import torch

@triton.jit
def s31111_kernel(a_ptr, sum_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid == 0:
        # Initialize sum
        sum_val = 0.0
        
        # sum += test(a) - sum first 4 elements starting at index 0
        offsets_0 = tl.arange(0, 4)
        mask_0 = offsets_0 < N
        vals_0 = tl.load(a_ptr + offsets_0, mask=mask_0, other=0.0)
        sum_val += tl.sum(vals_0)
        
        # sum += test(&a[4]) - sum 4 elements starting at index 4
        if N > 4:
            offsets_4 = 4 + tl.arange(0, 4)
            mask_4 = offsets_4 < N
            vals_4 = tl.load(a_ptr + offsets_4, mask=mask_4, other=0.0)
            sum_val += tl.sum(vals_4)
        
        # sum += test(&a[8]) - sum 4 elements starting at index 8
        if N > 8:
            offsets_8 = 8 + tl.arange(0, 4)
            mask_8 = offsets_8 < N
            vals_8 = tl.load(a_ptr + offsets_8, mask=mask_8, other=0.0)
            sum_val += tl.sum(vals_8)
        
        # sum += test(&a[12]) - sum 4 elements starting at index 12
        if N > 12:
            offsets_12 = 12 + tl.arange(0, 4)
            mask_12 = offsets_12 < N
            vals_12 = tl.load(a_ptr + offsets_12, mask=mask_12, other=0.0)
            sum_val += tl.sum(vals_12)
        
        # sum += test(&a[16]) - sum 4 elements starting at index 16
        if N > 16:
            offsets_16 = 16 + tl.arange(0, 4)
            mask_16 = offsets_16 < N
            vals_16 = tl.load(a_ptr + offsets_16, mask=mask_16, other=0.0)
            sum_val += tl.sum(vals_16)
        
        # sum += test(&a[20]) - sum 4 elements starting at index 20
        if N > 20:
            offsets_20 = 20 + tl.arange(0, 4)
            mask_20 = offsets_20 < N
            vals_20 = tl.load(a_ptr + offsets_20, mask=mask_20, other=0.0)
            sum_val += tl.sum(vals_20)
        
        # sum += test(&a[24]) - sum 4 elements starting at index 24
        if N > 24:
            offsets_24 = 24 + tl.arange(0, 4)
            mask_24 = offsets_24 < N
            vals_24 = tl.load(a_ptr + offsets_24, mask=mask_24, other=0.0)
            sum_val += tl.sum(vals_24)
        
        # sum += test(&a[28]) - sum 4 elements starting at index 28
        if N > 28:
            offsets_28 = 28 + tl.arange(0, 4)
            mask_28 = offsets_28 < N
            vals_28 = tl.load(a_ptr + offsets_28, mask=mask_28, other=0.0)
            sum_val += tl.sum(vals_28)
        
        # Store the final sum
        tl.store(sum_ptr, sum_val)

def s31111_triton(a):
    N = a.shape[0]
    
    # Create output tensor for sum
    sum_tensor = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 32
    grid = (1,)
    
    s31111_kernel[grid](
        a, sum_tensor, N, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return sum_tensor.item()