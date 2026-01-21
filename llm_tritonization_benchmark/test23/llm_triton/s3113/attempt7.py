import triton
import triton.language as tl
import torch

@triton.jit
def s3113_kernel(a_ptr, result_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize with first element's absolute value
    first_val = tl.load(a_ptr)
    max_val = tl.abs(first_val)
    
    # Process array in blocks starting from index 0
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        # Load values and compute absolute values
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        abs_vals = tl.abs(vals)
        
        # Update maximum with valid elements only
        for i in range(BLOCK_SIZE):
            if block_start + i < n:
                if i == 0 or abs_vals[i] > max_val:
                    if abs_vals[i] > max_val:
                        max_val = abs_vals[i]
    
    # Store result
    tl.store(result_ptr, max_val)

@triton.jit
def s3113_kernel_fixed(a_ptr, result_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid != 0:
        return
    
    # Initialize with first element
    max_val = tl.abs(tl.load(a_ptr))
    
    # Process all elements including first one
    offsets = tl.arange(0, BLOCK_SIZE)
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=-1e30)
        abs_vals = tl.abs(vals)
        
        # Set masked elements to -1 so they don't affect max
        abs_vals = tl.where(mask, abs_vals, -1.0)
        
        block_max = tl.max(abs_vals)
        max_val = tl.maximum(max_val, block_max)
    
    tl.store(result_ptr, max_val)

def s3113_triton(a):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    grid = (1,)
    s3113_kernel_fixed[grid](a, result, n, BLOCK_SIZE=BLOCK_SIZE)
    
    return result.item()