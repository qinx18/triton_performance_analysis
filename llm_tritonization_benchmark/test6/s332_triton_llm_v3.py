import torch
import triton
import triton.language as tl

@triton.jit
def s332_kernel(a_ptr, t, result_index_ptr, result_value_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    found_index = -2
    found_value = -1.0
    found = False
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
        
        condition = a_vals > t
        valid_condition = condition & mask
        
        if tl.any(valid_condition):
            # Find the first valid index in this block
            first_valid_mask = valid_condition & (tl.cumsum(valid_condition.to(tl.int32), axis=0) == 1)
            if tl.any(first_valid_mask):
                # Extract the index and value
                indices = tl.where(first_valid_mask, current_offsets, n_elements)
                values = tl.where(first_valid_mask, a_vals, -1.0)
                
                min_idx = tl.min(indices, axis=0)
                if min_idx < n_elements:
                    found_index = min_idx
                    found_value = tl.sum(tl.where(current_offsets == found_index, a_vals, 0.0), axis=0)
                    found = True
                    break
    
    if tl.program_id(0) == 0:
        tl.store(result_index_ptr, found_index)
        tl.store(result_value_ptr, found_value)

def s332_triton(a, t_val):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    result_index = torch.tensor([-2], dtype=torch.int32, device=a.device)
    result_value = torch.tensor([-1.0], dtype=a.dtype, device=a.device)
    
    grid = (1,)
    s332_kernel[grid](
        a, t_val, result_index, result_value, n_elements, BLOCK_SIZE
    )
    
    chksum = result_value.item() + float(result_index.item())
    return result_value.item(), result_index.item(), chksum