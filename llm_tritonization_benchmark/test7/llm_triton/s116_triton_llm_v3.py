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
    block_idx = tl.program_id(0)
    block_start = block_idx * BLOCK_SIZE * 5
    
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for step in range(0, BLOCK_SIZE, 1):
        i = block_start + step * 5
        
        if i + 5 < n_elements:
            # Load values from read-only copy
            a_i = tl.load(a_copy_ptr + i)
            a_i1 = tl.load(a_copy_ptr + i + 1)
            a_i2 = tl.load(a_copy_ptr + i + 2)
            a_i3 = tl.load(a_copy_ptr + i + 3)
            a_i4 = tl.load(a_copy_ptr + i + 4)
            a_i5 = tl.load(a_copy_ptr + i + 5)
            
            # Compute results
            result_0 = a_i1 * a_i
            result_1 = a_i2 * a_i1
            result_2 = a_i3 * a_i2
            result_3 = a_i4 * a_i3
            result_4 = a_i5 * a_i4
            
            # Store to original array
            tl.store(a_ptr + i, result_0)
            tl.store(a_ptr + i + 1, result_1)
            tl.store(a_ptr + i + 2, result_2)
            tl.store(a_ptr + i + 3, result_3)
            tl.store(a_ptr + i + 4, result_4)

def s116_triton(a):
    n_elements = a.size(0)
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    BLOCK_SIZE = 128
    num_blocks = triton.cdiv((n_elements - 5), 5 * BLOCK_SIZE)
    
    grid = (num_blocks,)
    
    s116_kernel[grid](
        a,
        a_copy,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )