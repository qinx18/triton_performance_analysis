import torch
import triton
import triton.language as tl

@triton.jit
def s313_kernel(a_ptr, b_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize accumulator
    dot_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Process blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load data with masking
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute products and accumulate
        products = a_vals * b_vals
        dot_acc += tl.where(mask, products, 0.0)
    
    # Reduce within block
    dot_sum = tl.sum(dot_acc)
    
    # Store result (first thread in block)
    if tl.program_id(0) == 0:
        tl.store(result_ptr, dot_sum)

def s313_triton(a, b):
    n_elements = a.shape[0]
    
    # Create output tensor
    result = torch.zeros(1, device=a.device, dtype=a.dtype)
    
    # Launch kernel with single block
    BLOCK_SIZE = 256
    grid = (1,)
    
    s313_kernel[grid](
        a, b, result,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()