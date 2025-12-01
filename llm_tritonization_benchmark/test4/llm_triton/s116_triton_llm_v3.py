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
    block_start = tl.program_id(0) * BLOCK_SIZE * 5
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_offset in range(0, BLOCK_SIZE, 1):
        base_idx = block_start + block_offset * 5
        
        if base_idx + 5 < n_elements:
            # Load values from read-only copy
            a_val_0 = tl.load(a_copy_ptr + base_idx)
            a_val_1 = tl.load(a_copy_ptr + base_idx + 1)
            a_val_2 = tl.load(a_copy_ptr + base_idx + 2)
            a_val_3 = tl.load(a_copy_ptr + base_idx + 3)
            a_val_4 = tl.load(a_copy_ptr + base_idx + 4)
            a_val_5 = tl.load(a_copy_ptr + base_idx + 5)
            
            # Compute results
            result_0 = a_val_1 * a_val_0
            result_1 = a_val_2 * a_val_1
            result_2 = a_val_3 * a_val_2
            result_3 = a_val_4 * a_val_3
            result_4 = a_val_5 * a_val_4
            
            # Store results to original array
            tl.store(a_ptr + base_idx, result_0)
            tl.store(a_ptr + base_idx + 1, result_1)
            tl.store(a_ptr + base_idx + 2, result_2)
            tl.store(a_ptr + base_idx + 3, result_3)
            tl.store(a_ptr + base_idx + 4, result_4)

def s116_triton(a):
    n_elements = a.shape[0]
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements - 5, BLOCK_SIZE * 5),)
    
    s116_kernel[grid](
        a,
        a_copy,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a