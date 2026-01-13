import torch
import triton
import triton.language as tl

@triton.jit
def s424_kernel(flat_2d_array_ptr, a_ptr, strip_start, strip_size, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + block_id * BLOCK_SIZE + offsets
    
    mask = offsets < strip_size - block_id * BLOCK_SIZE
    
    # Load from flat_2d_array[i] and a[i]
    flat_vals = tl.load(flat_2d_array_ptr + idx, mask=mask, other=0.0)
    a_vals = tl.load(a_ptr + idx, mask=mask, other=0.0)
    
    result = flat_vals + a_vals
    
    # Store to xx[i+1] which is flat_2d_array[i+64]
    tl.store(flat_2d_array_ptr + idx + 64, result, mask=mask)

def s424_triton(a, flat_2d_array):
    n_elements = a.shape[0] - 1  # LEN_1D - 1
    
    BLOCK_SIZE = 64
    STRIP_SIZE = 64
    
    # Process strips sequentially to avoid race conditions
    for strip_start in range(0, n_elements, STRIP_SIZE):
        strip_end = min(strip_start + STRIP_SIZE, n_elements)
        strip_size = strip_end - strip_start
        
        if strip_size <= 0:
            break
            
        # Launch kernel for this strip
        num_blocks = triton.cdiv(strip_size, BLOCK_SIZE)
        grid = (num_blocks,)
        s424_kernel[grid](
            flat_2d_array,
            a,
            strip_start,
            strip_size,
            BLOCK_SIZE=BLOCK_SIZE
        )