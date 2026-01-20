import torch
import triton
import triton.language as tl

@triton.jit
def s31111_kernel(a_ptr, sum_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Calculate sum of elements at specific offsets
    sum_val = 0.0
    
    # Define offsets for loading 4 consecutive elements
    offsets = tl.arange(0, 4)
    
    # sum += test(a) - sum first 4 elements starting at index 0
    mask0 = offsets < N
    vals0 = tl.load(a_ptr + offsets, mask=mask0, other=0.0)
    sum_val += tl.sum(vals0)
    
    # sum += test(&a[4]) - sum 4 elements starting at index 4
    offsets4 = 4 + offsets
    mask4 = offsets4 < N
    vals4 = tl.load(a_ptr + offsets4, mask=mask4, other=0.0)
    sum_val += tl.sum(vals4)
    
    # sum += test(&a[8]) - sum 4 elements starting at index 8
    offsets8 = 8 + offsets
    mask8 = offsets8 < N
    vals8 = tl.load(a_ptr + offsets8, mask=mask8, other=0.0)
    sum_val += tl.sum(vals8)
    
    # sum += test(&a[12]) - sum 4 elements starting at index 12
    offsets12 = 12 + offsets
    mask12 = offsets12 < N
    vals12 = tl.load(a_ptr + offsets12, mask=mask12, other=0.0)
    sum_val += tl.sum(vals12)
    
    # sum += test(&a[16]) - sum 4 elements starting at index 16
    offsets16 = 16 + offsets
    mask16 = offsets16 < N
    vals16 = tl.load(a_ptr + offsets16, mask=mask16, other=0.0)
    sum_val += tl.sum(vals16)
    
    # sum += test(&a[20]) - sum 4 elements starting at index 20
    offsets20 = 20 + offsets
    mask20 = offsets20 < N
    vals20 = tl.load(a_ptr + offsets20, mask=mask20, other=0.0)
    sum_val += tl.sum(vals20)
    
    # sum += test(&a[24]) - sum 4 elements starting at index 24
    offsets24 = 24 + offsets
    mask24 = offsets24 < N
    vals24 = tl.load(a_ptr + offsets24, mask=mask24, other=0.0)
    sum_val += tl.sum(vals24)
    
    # sum += test(&a[28]) - sum 4 elements starting at index 28
    offsets28 = 28 + offsets
    mask28 = offsets28 < N
    vals28 = tl.load(a_ptr + offsets28, mask=mask28, other=0.0)
    sum_val += tl.sum(vals28)
    
    # Store the final sum
    tl.store(sum_ptr, sum_val)

def s31111_triton(a):
    N = a.shape[0]
    
    # Create output tensor for the sum
    sum_result = torch.zeros(1, dtype=torch.float32, device=a.device)
    
    # Launch kernel with single thread since we're computing one sum
    grid = (1,)
    s31111_kernel[grid](a, sum_result, N, BLOCK_SIZE=32)
    
    return sum_result.item()