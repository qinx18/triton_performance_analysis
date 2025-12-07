import torch
import triton
import triton.language as tl

@triton.jit
def s31111_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    if tl.program_id(0) != 0:
        return
    
    sum_val = 0.0
    
    # Manually unroll the 8 test() calls with different starting offsets
    # test(a) - offset 0
    offsets_0 = tl.arange(0, 4)
    mask_0 = offsets_0 < n_elements
    vals_0 = tl.load(a_ptr + offsets_0, mask=mask_0, other=0.0)
    sum_val += tl.sum(vals_0)
    
    # test(&a[4]) - offset 4
    offsets_4 = tl.arange(0, 4) + 4
    mask_4 = offsets_4 < n_elements
    vals_4 = tl.load(a_ptr + offsets_4, mask=mask_4, other=0.0)
    sum_val += tl.sum(vals_4)
    
    # test(&a[8]) - offset 8
    offsets_8 = tl.arange(0, 4) + 8
    mask_8 = offsets_8 < n_elements
    vals_8 = tl.load(a_ptr + offsets_8, mask=mask_8, other=0.0)
    sum_val += tl.sum(vals_8)
    
    # test(&a[12]) - offset 12
    offsets_12 = tl.arange(0, 4) + 12
    mask_12 = offsets_12 < n_elements
    vals_12 = tl.load(a_ptr + offsets_12, mask=mask_12, other=0.0)
    sum_val += tl.sum(vals_12)
    
    # test(&a[16]) - offset 16
    offsets_16 = tl.arange(0, 4) + 16
    mask_16 = offsets_16 < n_elements
    vals_16 = tl.load(a_ptr + offsets_16, mask=mask_16, other=0.0)
    sum_val += tl.sum(vals_16)
    
    # test(&a[20]) - offset 20
    offsets_20 = tl.arange(0, 4) + 20
    mask_20 = offsets_20 < n_elements
    vals_20 = tl.load(a_ptr + offsets_20, mask=mask_20, other=0.0)
    sum_val += tl.sum(vals_20)
    
    # test(&a[24]) - offset 24
    offsets_24 = tl.arange(0, 4) + 24
    mask_24 = offsets_24 < n_elements
    vals_24 = tl.load(a_ptr + offsets_24, mask=mask_24, other=0.0)
    sum_val += tl.sum(vals_24)
    
    # test(&a[28]) - offset 28
    offsets_28 = tl.arange(0, 4) + 28
    mask_28 = offsets_28 < n_elements
    vals_28 = tl.load(a_ptr + offsets_28, mask=mask_28, other=0.0)
    sum_val += tl.sum(vals_28)
    
    tl.store(result_ptr, sum_val)

def s31111_triton(a):
    n_elements = a.numel()
    BLOCK_SIZE = 32
    
    result = torch.zeros(1, device=a.device, dtype=a.dtype)
    
    grid = (1,)
    s31111_kernel[grid](
        a, result, n_elements, BLOCK_SIZE
    )
    
    return result.item()