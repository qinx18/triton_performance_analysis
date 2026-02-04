import triton
import triton.language as tl
import torch

@triton.jit
def s319_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, sum_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize sum accumulator
    sum_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load c, d, e arrays
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
        e_vals = tl.load(e_ptr + current_offsets, mask=mask, other=0.0)
        
        # a[i] = c[i] + d[i]
        a_vals = c_vals + d_vals
        tl.store(a_ptr + current_offsets, a_vals, mask=mask)
        
        # sum += a[i]
        sum_acc += tl.where(mask, a_vals, 0.0)
        
        # b[i] = c[i] + e[i]
        b_vals = c_vals + e_vals
        tl.store(b_ptr + current_offsets, b_vals, mask=mask)
        
        # sum += b[i]
        sum_acc += tl.where(mask, b_vals, 0.0)
    
    # Reduce sum across the block
    total_sum = tl.sum(sum_acc)
    
    # Store the sum (only first thread in block)
    if tl.program_id(0) == 0 and offsets[0] == 0:
        tl.atomic_add(sum_ptr, total_sum)

def s319_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Initialize sum result
    sum_result = torch.zeros(1, dtype=torch.float32, device=a.device)
    
    # Launch kernel with single program
    grid = (1,)
    s319_kernel[grid](a, b, c, d, e, sum_result, n_elements, BLOCK_SIZE)
    
    return sum_result.item()