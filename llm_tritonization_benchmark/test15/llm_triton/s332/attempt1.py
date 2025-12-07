import torch
import triton
import triton.language as tl

@triton.jit
def s332_kernel(a_ptr, t, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
        
        # Check if any element is greater than t
        condition = a_vals > t
        if tl.any(condition):
            # Find first index where condition is true
            for i in range(BLOCK_SIZE):
                offset_idx = block_start + i
                if offset_idx < n_elements:
                    val = tl.load(a_ptr + offset_idx)
                    if val > t:
                        tl.store(result_ptr, val)
                        tl.store(result_ptr + 1, offset_idx)
                        return

def s332_triton(a, t):
    n_elements = a.numel()
    
    # Create result tensor [value, index]
    result = torch.tensor([-1.0, -2.0], device=a.device, dtype=a.dtype)
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s332_kernel[grid](
        a, t, result, n_elements, BLOCK_SIZE
    )
    
    value = result[0].item()
    index = int(result[1].item())
    
    # Compute chksum (though not returned)
    chksum = value + float(index)
    
    return value