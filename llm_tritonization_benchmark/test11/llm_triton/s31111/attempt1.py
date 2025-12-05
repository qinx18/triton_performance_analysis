import torch
import triton
import triton.language as tl

@triton.jit
def s31111_kernel(a_ptr, sum_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel computes the sum reduction by accessing specific positions in array 'a'
    # and summing 4 consecutive elements starting from each position
    
    program_id = tl.program_id(0)
    
    if program_id == 0:
        # Load elements for test(a) - indices 0,1,2,3
        offsets_0 = tl.arange(0, 4)
        mask_0 = offsets_0 < n_elements
        vals_0 = tl.load(a_ptr + offsets_0, mask=mask_0, other=0.0)
        sum_0 = tl.sum(vals_0)
        
        # Load elements for test(&a[4]) - indices 4,5,6,7
        offsets_4 = 4 + tl.arange(0, 4)
        mask_4 = offsets_4 < n_elements
        vals_4 = tl.load(a_ptr + offsets_4, mask=mask_4, other=0.0)
        sum_4 = tl.sum(vals_4)
        
        # Load elements for test(&a[8]) - indices 8,9,10,11
        offsets_8 = 8 + tl.arange(0, 4)
        mask_8 = offsets_8 < n_elements
        vals_8 = tl.load(a_ptr + offsets_8, mask=mask_8, other=0.0)
        sum_8 = tl.sum(vals_8)
        
        # Load elements for test(&a[12]) - indices 12,13,14,15
        offsets_12 = 12 + tl.arange(0, 4)
        mask_12 = offsets_12 < n_elements
        vals_12 = tl.load(a_ptr + offsets_12, mask=mask_12, other=0.0)
        sum_12 = tl.sum(vals_12)
        
        # Load elements for test(&a[16]) - indices 16,17,18,19
        offsets_16 = 16 + tl.arange(0, 4)
        mask_16 = offsets_16 < n_elements
        vals_16 = tl.load(a_ptr + offsets_16, mask=mask_16, other=0.0)
        sum_16 = tl.sum(vals_16)
        
        # Load elements for test(&a[20]) - indices 20,21,22,23
        offsets_20 = 20 + tl.arange(0, 4)
        mask_20 = offsets_20 < n_elements
        vals_20 = tl.load(a_ptr + offsets_20, mask=mask_20, other=0.0)
        sum_20 = tl.sum(vals_20)
        
        # Load elements for test(&a[24]) - indices 24,25,26,27
        offsets_24 = 24 + tl.arange(0, 4)
        mask_24 = offsets_24 < n_elements
        vals_24 = tl.load(a_ptr + offsets_24, mask=mask_24, other=0.0)
        sum_24 = tl.sum(vals_24)
        
        # Load elements for test(&a[28]) - indices 28,29,30,31
        offsets_28 = 28 + tl.arange(0, 4)
        mask_28 = offsets_28 < n_elements
        vals_28 = tl.load(a_ptr + offsets_28, mask=mask_28, other=0.0)
        sum_28 = tl.sum(vals_28)
        
        # Calculate total sum
        total_sum = sum_0 + sum_4 + sum_8 + sum_12 + sum_16 + sum_20 + sum_24 + sum_28
        
        # Store result
        tl.store(sum_ptr, total_sum)

def s31111_triton(a):
    n_elements = a.numel()
    
    # Allocate output tensor for the sum
    sum_result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single block since we're doing a single reduction
    BLOCK_SIZE = 32
    grid = (1,)
    
    s31111_kernel[grid](
        a, sum_result, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return sum_result.item()