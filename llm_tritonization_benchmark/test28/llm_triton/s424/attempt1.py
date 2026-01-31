import torch
import triton
import triton.language as tl

@triton.jit
def s424_kernel(flat_2d_array, a, vl: tl.constexpr, strip_start, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets
    
    mask = idx < n_elements
    
    # Read from flat_2d_array[idx]
    read_vals = tl.load(flat_2d_array + idx, mask=mask, other=0.0)
    
    # Read from a[idx]  
    a_vals = tl.load(a + idx, mask=mask, other=0.0)
    
    # Compute sum
    result = read_vals + a_vals
    
    # Write to xx[idx+1] which is flat_2d_array[idx + vl + 1]
    write_idx = idx + vl + 1
    tl.store(flat_2d_array + write_idx, result, mask=mask)

def s424_triton(a, flat_2d_array, xx):
    vl = 63
    n_elements = a.shape[0] - 1  # LEN_1D - 1
    
    STRIP_SIZE = 64
    BLOCK_SIZE = 64
    
    # Process in sequential strips due to RAW dependency
    for strip_start in range(0, n_elements, STRIP_SIZE):
        strip_end = min(strip_start + STRIP_SIZE, n_elements)
        strip_len = strip_end - strip_start
        
        # Launch one thread block to process this strip
        s424_kernel[(1,)](
            flat_2d_array,
            a, 
            vl,
            strip_start,
            strip_end,
            BLOCK_SIZE=BLOCK_SIZE
        )