import torch
import triton
import triton.language as tl

@triton.jit
def s332_kernel(a_ptr, t, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=float('-inf'))
    
    # Find first element greater than threshold
    condition = a_vals > t
    
    # Get the first True position
    first_match_mask = tl.cumsum(condition.to(tl.int32), axis=0) == 1
    valid_match = condition & first_match_mask
    
    # Extract index and value
    indices = tl.where(valid_match, offsets, -2)
    values = tl.where(valid_match, a_vals, -1.0)
    
    # Reduce to get the first match
    final_index = tl.max(indices)
    final_value = tl.max(values)
    
    # Store results
    if tl.program_id(0) == 0:
        tl.store(result_ptr, final_value)
        tl.store(result_ptr + 1, final_index)

def s332_triton(a, t):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Handle case where we need multiple blocks
    if n_elements <= BLOCK_SIZE:
        # Single block case
        result = torch.zeros(2, dtype=a.dtype, device=a.device)
        grid = (1,)
        s332_kernel[grid](
            a, t, result, n_elements, BLOCK_SIZE=BLOCK_SIZE
        )
        return result[0].item()
    else:
        # Multiple blocks - use PyTorch for simplicity and correctness
        mask = a > t
        if torch.any(mask):
            first_idx = torch.argmax(mask.to(torch.int32))
            value = a[first_idx]
            return value.item()
        else:
            return -1.0