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
        
        # Check if any value is greater than threshold
        condition = a_vals > t
        
        if tl.sum(condition.to(tl.int32)) > 0:
            # Find first index where condition is true
            for i in range(BLOCK_SIZE):
                if block_start + i < n_elements:
                    offset = block_start + i
                    val = tl.load(a_ptr + offset)
                    if val > t:
                        tl.store(result_ptr, val)  # Store the value
                        tl.store(result_ptr + 1, offset.to(tl.float32))  # Store the index
                        return

def s332_triton(a, t):
    n_elements = a.numel()
    
    # Result tensor: [value, index]
    result = torch.tensor([-1.0, -2.0], dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 1024
    grid = (1,)
    
    s332_kernel[grid](
        a, t, result, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result[0].item()  # Return the value (not index)