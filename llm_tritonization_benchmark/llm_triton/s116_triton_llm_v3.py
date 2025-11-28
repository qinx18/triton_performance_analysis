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
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE * 5
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE) * 5
    mask = offsets < (n_elements - 5)
    
    # Load values from read-only copy
    a_i = tl.load(a_copy_ptr + offsets, mask=mask)
    a_i_plus_1 = tl.load(a_copy_ptr + offsets + 1, mask=mask)
    a_i_plus_2 = tl.load(a_copy_ptr + offsets + 2, mask=mask)
    a_i_plus_3 = tl.load(a_copy_ptr + offsets + 3, mask=mask)
    a_i_plus_4 = tl.load(a_copy_ptr + offsets + 4, mask=mask)
    a_i_plus_5 = tl.load(a_copy_ptr + offsets + 5, mask=mask)
    
    # Compute results
    result_0 = a_i_plus_1 * a_i
    result_1 = a_i_plus_2 * a_i_plus_1
    result_2 = a_i_plus_3 * a_i_plus_2
    result_3 = a_i_plus_4 * a_i_plus_3
    result_4 = a_i_plus_5 * a_i_plus_4
    
    # Store to original array
    tl.store(a_ptr + offsets, result_0, mask=mask)
    tl.store(a_ptr + offsets + 1, result_1, mask=mask)
    tl.store(a_ptr + offsets + 2, result_2, mask=mask)
    tl.store(a_ptr + offsets + 3, result_3, mask=mask)
    tl.store(a_ptr + offsets + 4, result_4, mask=mask)

def s116_triton(a):
    n_elements = a.shape[0]
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    BLOCK_SIZE = 128
    grid = ((n_elements + BLOCK_SIZE * 5 - 1) // (BLOCK_SIZE * 5),)
    
    s116_kernel[grid](
        a,
        a_copy,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a