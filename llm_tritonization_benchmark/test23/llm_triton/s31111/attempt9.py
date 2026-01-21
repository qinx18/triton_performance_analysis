import triton
import triton.language as tl
import torch

@triton.jit
def s31111_kernel(a_ptr, sum_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if pid > 0:
        return
    
    # Initialize sum
    total_sum = 0.0
    
    # Process each test call - sum 4 elements starting at each offset
    offsets_4 = tl.arange(0, 4)
    
    # Sum elements at indices [0,1,2,3]
    indices_0 = 0 + offsets_4
    mask_0 = indices_0 < n_elements
    vals_0 = tl.load(a_ptr + indices_0, mask=mask_0, other=0.0)
    total_sum += tl.sum(vals_0)
    
    # Sum elements at indices [4,5,6,7]
    indices_4 = 4 + offsets_4
    mask_4 = indices_4 < n_elements
    vals_4 = tl.load(a_ptr + indices_4, mask=mask_4, other=0.0)
    total_sum += tl.sum(vals_4)
    
    # Sum elements at indices [8,9,10,11]
    indices_8 = 8 + offsets_4
    mask_8 = indices_8 < n_elements
    vals_8 = tl.load(a_ptr + indices_8, mask=mask_8, other=0.0)
    total_sum += tl.sum(vals_8)
    
    # Sum elements at indices [12,13,14,15]
    indices_12 = 12 + offsets_4
    mask_12 = indices_12 < n_elements
    vals_12 = tl.load(a_ptr + indices_12, mask=mask_12, other=0.0)
    total_sum += tl.sum(vals_12)
    
    # Sum elements at indices [16,17,18,19]
    indices_16 = 16 + offsets_4
    mask_16 = indices_16 < n_elements
    vals_16 = tl.load(a_ptr + indices_16, mask=mask_16, other=0.0)
    total_sum += tl.sum(vals_16)
    
    # Sum elements at indices [20,21,22,23]
    indices_20 = 20 + offsets_4
    mask_20 = indices_20 < n_elements
    vals_20 = tl.load(a_ptr + indices_20, mask=mask_20, other=0.0)
    total_sum += tl.sum(vals_20)
    
    # Sum elements at indices [24,25,26,27]
    indices_24 = 24 + offsets_4
    mask_24 = indices_24 < n_elements
    vals_24 = tl.load(a_ptr + indices_24, mask=mask_24, other=0.0)
    total_sum += tl.sum(vals_24)
    
    # Sum elements at indices [28,29,30,31]
    indices_28 = 28 + offsets_4
    mask_28 = indices_28 < n_elements
    vals_28 = tl.load(a_ptr + indices_28, mask=mask_28, other=0.0)
    total_sum += tl.sum(vals_28)
    
    # Store the result
    tl.store(sum_ptr, total_sum)

def s31111_triton(a):
    n_elements = a.shape[0]
    
    # Create output tensor for the sum
    sum_result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 128
    grid = (1,)
    
    s31111_kernel[grid](
        a, sum_result, n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return sum_result.item()