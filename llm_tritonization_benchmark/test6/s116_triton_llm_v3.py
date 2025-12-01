import triton
import triton.language as tl
import torch

@triton.jit
def s116_kernel(
    a_ptr,
    a_copy_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements - 5, BLOCK_SIZE * 5):
        # Process 5 elements per iteration
        for step in range(5):
            current_offsets = block_start + step * BLOCK_SIZE + offsets
            mask = current_offsets < (n_elements - 5)
            
            # Load values from read-only copy
            a_vals = tl.load(a_copy_ptr + current_offsets, mask=mask)
            a_next_vals = tl.load(a_copy_ptr + current_offsets + 1, mask=mask)
            
            # Compute result
            result = a_next_vals * a_vals
            
            # Store to original array
            tl.store(a_ptr + current_offsets, result, mask=mask)

def s116_triton(a):
    n_elements = a.numel()
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s116_kernel[grid](
        a,
        a_copy,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )