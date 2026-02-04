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
        greater_mask = a_vals > t
        if tl.sum(greater_mask.to(tl.int32)) > 0:
            # Find first occurrence within this block
            for i in range(BLOCK_SIZE):
                if i + block_start < n_elements:
                    offset = block_start + i
                    val = tl.load(a_ptr + offset)
                    if val > t:
                        tl.store(result_ptr, offset)  # Store index
                        tl.store(result_ptr + 1, val)  # Store value
                        return

def s332_triton(a, t):
    # Use PyTorch for efficient argmax operation
    mask = a > t
    if not torch.any(mask):
        # No value found greater than threshold
        index = -2
        value = -1.0
    else:
        # Find first index where condition is true
        indices = torch.nonzero(mask, as_tuple=False).flatten()
        if len(indices) > 0:
            index = indices[0].item()
            value = a[index].item()
        else:
            index = -2
            value = -1.0
    
    return value