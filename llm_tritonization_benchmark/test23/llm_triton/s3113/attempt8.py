import triton
import triton.language as tl
import torch

@triton.jit
def s3113_kernel(a_ptr, result_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid != 0:
        return
    
    # Initialize with first element's absolute value
    max_val = tl.abs(tl.load(a_ptr))
    
    # Process all elements starting from index 0
    offsets = tl.arange(0, BLOCK_SIZE)
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        # Load values and compute absolute values
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        abs_vals = tl.abs(vals)
        
        # For each valid element, check if it's greater than current max
        for i in range(BLOCK_SIZE):
            if block_start + i < n:
                current_abs = tl.abs(tl.load(a_ptr + block_start + i))
                max_val = tl.where(current_abs > max_val, current_abs, max_val)
    
    tl.store(result_ptr, max_val)

def s3113_triton(a):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    grid = (1,)
    s3113_kernel[grid](a, result, n, BLOCK_SIZE=BLOCK_SIZE)
    
    return result.item()