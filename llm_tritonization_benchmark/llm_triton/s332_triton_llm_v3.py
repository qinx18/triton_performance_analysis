import torch
import triton
import triton.language as tl

@triton.jit
def s332_kernel(a_ptr, result_ptr, t, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    index = -2
    value = -1.0
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
        
        # Check if any element is greater than threshold
        condition = a_vals > t
        has_match = tl.sum(condition.to(tl.int32)) > 0
        
        if has_match:
            # Find first element that matches condition
            for i in range(BLOCK_SIZE):
                offset = block_start + i
                if offset < n_elements:
                    val = tl.load(a_ptr + offset)
                    if val > t:
                        index = offset
                        value = val
                        break
            break
    
    chksum = value + index
    tl.store(result_ptr, chksum)

def s332_triton(a, t):
    n_elements = a.numel()
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 256
    
    s332_kernel[(1,)](
        a, result, t, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()