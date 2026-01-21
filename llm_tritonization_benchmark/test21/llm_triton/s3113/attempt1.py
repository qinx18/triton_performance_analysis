import triton
import triton.language as tl
import torch

@triton.jit
def s3113_kernel(a_ptr, result_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize global max with absolute value of first element
    if tl.program_id(0) == 0:
        first_val = tl.load(a_ptr)
        tl.store(result_ptr, tl.abs(first_val))
    
    # Process blocks
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        abs_vals = tl.abs(vals)
        block_max = tl.max(abs_vals)
        
        # Atomic max with current global max
        current_global_max = tl.load(result_ptr)
        if block_max > current_global_max:
            tl.store(result_ptr, block_max)

def s3113_triton(a):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Result tensor to store the maximum absolute value
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single thread block for reduction
    grid = (1,)
    s3113_kernel[grid](a, result, n, BLOCK_SIZE=BLOCK_SIZE)
    
    return result.item()