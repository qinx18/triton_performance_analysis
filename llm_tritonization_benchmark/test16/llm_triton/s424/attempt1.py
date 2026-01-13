import torch
import triton
import triton.language as tl

@triton.jit
def s424_kernel(flat_2d_array_ptr, a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n_elements
    
    # Load from flat_2d_array[i] and a[i]
    flat_vals = tl.load(flat_2d_array_ptr + idx, mask=mask, other=0.0)
    a_vals = tl.load(a_ptr + idx, mask=mask, other=0.0)
    
    result = flat_vals + a_vals
    
    # Store to flat_2d_array[i+64] (xx[i+1] where xx = flat_2d_array + 63)
    tl.store(flat_2d_array_ptr + idx + 64, result, mask=mask)

def s424_triton(a, flat_2d_array):
    n_elements = a.shape[0] - 1  # LEN_1D - 1
    
    BLOCK_SIZE = 64
    num_strips = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Process strips sequentially to avoid race conditions
    for strip_id in range(num_strips):
        strip_start = strip_id * BLOCK_SIZE
        strip_end = min(strip_start + BLOCK_SIZE, n_elements)
        
        if strip_start >= n_elements:
            break
            
        # Launch kernel for this strip only
        grid = (1,)
        s424_kernel[grid](
            flat_2d_array,
            a,
            strip_end - strip_start,
            BLOCK_SIZE=BLOCK_SIZE
        )