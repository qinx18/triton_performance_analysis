import triton
import triton.language as tl
import torch

@triton.jit
def s424_kernel(flat_2d_array_ptr, a_ptr, xx_ptr, strip_start, strip_size, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets
    
    mask = offsets < strip_size
    
    # Read from flat_2d_array[i] and a[i]
    flat_vals = tl.load(flat_2d_array_ptr + idx, mask=mask)
    a_vals = tl.load(a_ptr + idx, mask=mask)
    
    # Compute result
    result = flat_vals + a_vals
    
    # Write to xx[i+1]
    write_idx = idx + 1
    tl.store(xx_ptr + write_idx, result, mask=mask)

def s424_triton(a, flat_2d_array, xx):
    N = a.shape[0]
    n_elements = N - 1
    
    STRIP_SIZE = 64
    
    num_strips = triton.cdiv(n_elements, STRIP_SIZE)
    
    for strip_id in range(num_strips):
        strip_start = strip_id * STRIP_SIZE
        current_strip_size = min(STRIP_SIZE, n_elements - strip_start)
        
        if current_strip_size > 0:
            BLOCK_SIZE = triton.next_power_of_2(current_strip_size)
            
            s424_kernel[(1,)](
                flat_2d_array, a, xx,
                strip_start, current_strip_size,
                BLOCK_SIZE=BLOCK_SIZE
            )