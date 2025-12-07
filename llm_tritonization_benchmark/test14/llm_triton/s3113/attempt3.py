import torch
import triton
import triton.language as tl

@triton.jit
def s3113_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Load first element to initialize max
    first_val = tl.load(a_ptr)
    current_max = tl.abs(first_val)
    
    # Process all elements in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values and compute absolute values
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        abs_vals = tl.abs(vals)
        
        # Find maximum in this block and update global max
        local_max = tl.max(abs_vals, axis=0)
        current_max = tl.maximum(current_max, local_max)
    
    # Store result (only first thread stores)
    if tl.program_id(0) == 0:
        tl.store(result_ptr, current_max)

def s3113_triton(a, abs):
    n_elements = a.numel()
    
    # Create output tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single program
    BLOCK_SIZE = triton.next_power_of_2(min(n_elements, 1024))
    
    s3113_kernel[(1,)](
        a, result, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()