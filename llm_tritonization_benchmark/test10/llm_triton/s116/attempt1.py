import triton
import triton.language as tl
import torch

@triton.jit
def s116_kernel(a_ptr, a_copy_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements - 5, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = (current_offsets < n_elements - 5) & ((current_offsets % 5) == 0)
        
        # Load values from read-only copy
        a_i = tl.load(a_copy_ptr + current_offsets, mask=mask)
        a_i1 = tl.load(a_copy_ptr + current_offsets + 1, mask=mask)
        a_i2 = tl.load(a_copy_ptr + current_offsets + 2, mask=mask)
        a_i3 = tl.load(a_copy_ptr + current_offsets + 3, mask=mask)
        a_i4 = tl.load(a_copy_ptr + current_offsets + 4, mask=mask)
        a_i5 = tl.load(a_copy_ptr + current_offsets + 5, mask=mask)
        
        # Compute results
        result_0 = a_i1 * a_i
        result_1 = a_i2 * a_i1
        result_2 = a_i3 * a_i2
        result_3 = a_i4 * a_i3
        result_4 = a_i5 * a_i4
        
        # Store results to original array
        tl.store(a_ptr + current_offsets, result_0, mask=mask)
        tl.store(a_ptr + current_offsets + 1, result_1, mask=mask)
        tl.store(a_ptr + current_offsets + 2, result_2, mask=mask)
        tl.store(a_ptr + current_offsets + 3, result_3, mask=mask)
        tl.store(a_ptr + current_offsets + 4, result_4, mask=mask)

def s116_triton(a):
    n_elements = a.shape[0]
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements - 5, BLOCK_SIZE),)
    
    s116_kernel[grid](
        a,
        a_copy,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )