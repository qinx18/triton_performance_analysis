import triton
import triton.language as tl
import torch

@triton.jit
def s31111_kernel(a_ptr, result_ptr, BLOCK_SIZE: tl.constexpr):
    # Load the required 32 elements
    offsets = tl.arange(0, BLOCK_SIZE)
    a_vals = tl.load(a_ptr + offsets)
    
    # Compute test(a) - sum of a[0:4]
    offsets_0_4 = tl.arange(0, 4)
    sum_val = tl.sum(tl.load(a_ptr + offsets_0_4))
    
    # Compute test(&a[4]) - sum of a[4:8]
    offsets_4_8 = 4 + tl.arange(0, 4)
    sum_val += tl.sum(tl.load(a_ptr + offsets_4_8))
    
    # Compute test(&a[8]) - sum of a[8:12]
    offsets_8_12 = 8 + tl.arange(0, 4)
    sum_val += tl.sum(tl.load(a_ptr + offsets_8_12))
    
    # Compute test(&a[12]) - sum of a[12:16]
    offsets_12_16 = 12 + tl.arange(0, 4)
    sum_val += tl.sum(tl.load(a_ptr + offsets_12_16))
    
    # Compute test(&a[16]) - sum of a[16:20]
    offsets_16_20 = 16 + tl.arange(0, 4)
    sum_val += tl.sum(tl.load(a_ptr + offsets_16_20))
    
    # Compute test(&a[20]) - sum of a[20:24]
    offsets_20_24 = 20 + tl.arange(0, 4)
    sum_val += tl.sum(tl.load(a_ptr + offsets_20_24))
    
    # Compute test(&a[24]) - sum of a[24:28]
    offsets_24_28 = 24 + tl.arange(0, 4)
    sum_val += tl.sum(tl.load(a_ptr + offsets_24_28))
    
    # Compute test(&a[28]) - sum of a[28:32]
    offsets_28_32 = 28 + tl.arange(0, 4)
    sum_val += tl.sum(tl.load(a_ptr + offsets_28_32))
    
    # Store the result
    tl.store(result_ptr, sum_val)

def s31111_triton(a):
    # Create output tensor for the sum result
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single block since we just need one sum computation
    grid = (1,)
    s31111_kernel[grid](a, result, BLOCK_SIZE=32)
    
    return result[0].item()