import triton
import triton.language as tl
import torch

@triton.jit
def s116_kernel(a_ptr, a_copy_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for i in range(0, n_elements - 5, 5):
        if i >= block_start and i < block_start + BLOCK_SIZE:
            local_i = i - block_start
            
            if local_i < BLOCK_SIZE and i + 5 < n_elements:
                # Load values from read-only copy
                a_i = tl.load(a_copy_ptr + i)
                a_i_plus_1 = tl.load(a_copy_ptr + i + 1)
                a_i_plus_2 = tl.load(a_copy_ptr + i + 2)
                a_i_plus_3 = tl.load(a_copy_ptr + i + 3)
                a_i_plus_4 = tl.load(a_copy_ptr + i + 4)
                a_i_plus_5 = tl.load(a_copy_ptr + i + 5)
                
                # Compute and store to original array
                tl.store(a_ptr + i, a_i_plus_1 * a_i)
                tl.store(a_ptr + i + 1, a_i_plus_2 * a_i_plus_1)
                tl.store(a_ptr + i + 2, a_i_plus_3 * a_i_plus_2)
                tl.store(a_ptr + i + 3, a_i_plus_4 * a_i_plus_3)
                tl.store(a_ptr + i + 4, a_i_plus_5 * a_i_plus_4)

def s116_triton(a):
    n_elements = a.shape[0]
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s116_kernel[grid](
        a, a_copy, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )