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
        # Process 5 elements at a time
        for j in range(5):
            i_offset = block_start + j + offsets * 5
            mask = i_offset < n_elements - 5
            
            # Read from copy for both operands
            a_i = tl.load(a_copy_ptr + i_offset, mask=mask)
            a_i_plus_1 = tl.load(a_copy_ptr + i_offset + 1, mask=mask)
            
            # Compute and store to original
            result = a_i_plus_1 * a_i
            tl.store(a_ptr + i_offset, result, mask=mask)

def s116_triton(a):
    n_elements = a.numel()
    
    # Create read-only copy for WAR dependency handling
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(n_elements, BLOCK_SIZE * 5),)
    
    s116_kernel[grid](
        a,
        a_copy,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )