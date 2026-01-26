import triton
import triton.language as tl
import torch

@triton.jit
def s424_kernel(flat_2d_array_ptr, a_ptr, xx_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    strip_id = tl.program_id(0)
    strip_start = strip_id * 64
    
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets
    
    mask = idx < n_elements
    
    # Read from flat_2d_array[i] and a[i]
    flat_vals = tl.load(flat_2d_array_ptr + idx, mask=mask)
    a_vals = tl.load(a_ptr + idx, mask=mask)
    
    # Compute result
    result = flat_vals + a_vals
    
    # Write to xx[i+1] which is flat_2d_array[i+1+63] = flat_2d_array[i+64]
    tl.store(xx_ptr + idx + 1, result, mask=mask)

def s424_triton(a, flat_2d_array, xx):
    N = flat_2d_array.shape[0]
    n_elements = N - 1
    
    STRIP_SIZE = 64
    BLOCK_SIZE = 64
    
    num_strips = triton.cdiv(n_elements, STRIP_SIZE)
    
    for strip_id in range(num_strips):
        strip_start = strip_id * STRIP_SIZE
        strip_end = min(strip_start + STRIP_SIZE, n_elements)
        current_block_size = strip_end - strip_start
        
        if current_block_size > 0:
            actual_block_size = min(BLOCK_SIZE, triton.next_power_of_2(current_block_size))
            
            s424_kernel[(1,)](
                flat_2d_array, a, xx, 
                strip_end,
                BLOCK_SIZE=actual_block_size
            )