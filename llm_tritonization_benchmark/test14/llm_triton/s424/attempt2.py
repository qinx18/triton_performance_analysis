import torch
import triton
import triton.language as tl

@triton.jit
def s424_kernel(flat_2d_array_ptr, a_ptr, strip_start, strip_size, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets
    mask = offsets < strip_size
    
    # Read from flat_2d_array[i] and a[i]
    flat_vals = tl.load(flat_2d_array_ptr + idx, mask=mask)
    a_vals = tl.load(a_ptr + idx, mask=mask)
    
    # Compute result
    result = flat_vals + a_vals
    
    # Store to flat_2d_array[i+64] (xx[i+1] where xx = flat_2d_array + 63)
    store_idx = idx + 64
    tl.store(flat_2d_array_ptr + store_idx, result, mask=mask)

def s424_triton(a, flat_2d_array):
    LEN_1D = a.shape[0]
    n_elements = LEN_1D - 1
    
    STRIP_SIZE = 64
    BLOCK_SIZE = 64
    
    num_strips = triton.cdiv(n_elements, STRIP_SIZE)
    
    # Process strips sequentially to handle dependency
    for strip_id in range(num_strips):
        strip_start = strip_id * STRIP_SIZE
        strip_len = min(STRIP_SIZE, n_elements - strip_start)
        
        if strip_len > 0:
            s424_kernel[(1,)](
                flat_2d_array,
                a,
                strip_start,
                strip_len,
                BLOCK_SIZE=BLOCK_SIZE
            )