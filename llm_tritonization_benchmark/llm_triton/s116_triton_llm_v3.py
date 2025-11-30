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
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE * 5
    
    for block_offset in range(0, BLOCK_SIZE * 5, 5):
        i = block_start + block_offset
        
        if i + 5 < n_elements:
            # Load values from read-only copy
            a_i = tl.load(a_copy_ptr + i)
            a_i_1 = tl.load(a_copy_ptr + i + 1)
            a_i_2 = tl.load(a_copy_ptr + i + 2)
            a_i_3 = tl.load(a_copy_ptr + i + 3)
            a_i_4 = tl.load(a_copy_ptr + i + 4)
            a_i_5 = tl.load(a_copy_ptr + i + 5)
            
            # Compute results
            result_0 = a_i_1 * a_i
            result_1 = a_i_2 * a_i_1
            result_2 = a_i_3 * a_i_2
            result_3 = a_i_4 * a_i_3
            result_4 = a_i_5 * a_i_4
            
            # Store to original array
            tl.store(a_ptr + i, result_0)
            tl.store(a_ptr + i + 1, result_1)
            tl.store(a_ptr + i + 2, result_2)
            tl.store(a_ptr + i + 3, result_3)
            tl.store(a_ptr + i + 4, result_4)

def s116_triton(a):
    n_elements = a.shape[0]
    a_copy = a.clone()
    
    BLOCK_SIZE = 128
    grid = lambda meta: (triton.cdiv(n_elements - 5, BLOCK_SIZE * 5),)
    
    s116_kernel[grid](
        a,
        a_copy,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )