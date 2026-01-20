import torch
import triton
import triton.language as tl

@triton.jit
def s424_kernel(flat_2d_array_ptr, a_ptr, n_elements, STRIP_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    strip_id = tl.program_id(0)
    strip_start = strip_id * STRIP_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets
    
    # Mask for valid indices
    mask = (idx < n_elements) & (idx >= 0)
    
    # Load from flat_2d_array[i] and a[i]
    flat_vals = tl.load(flat_2d_array_ptr + idx, mask=mask, other=0.0)
    a_vals = tl.load(a_ptr + idx, mask=mask, other=0.0)
    
    # Compute result
    result = flat_vals + a_vals
    
    # Store to xx[i+1] which is flat_2d_array[i+64]
    write_idx = idx + 64
    write_mask = (idx < n_elements) & (idx >= 0) & ((write_idx) < (n_elements + 64))
    tl.store(flat_2d_array_ptr + write_idx, result, mask=write_mask)

def s424_triton(a, flat_2d_array):
    n_elements = a.shape[0] - 1  # LEN_1D - 1
    vl = 63
    STRIP_SIZE = 64
    BLOCK_SIZE = 64
    
    num_strips = triton.cdiv(n_elements, STRIP_SIZE)
    
    # Process strips sequentially to maintain dependency
    for strip_id in range(num_strips):
        strip_start = strip_id * STRIP_SIZE
        strip_end = min(strip_start + STRIP_SIZE, n_elements)
        strip_len = strip_end - strip_start
        
        if strip_len <= 0:
            break
            
        # Launch single block for this strip
        grid = (1,)
        s424_kernel[grid](
            flat_2d_array, 
            a, 
            strip_len,
            STRIP_SIZE=STRIP_SIZE,
            BLOCK_SIZE=min(BLOCK_SIZE, strip_len)
        )
        
        # Adjust pointers for next strip
        if strip_id < num_strips - 1:
            flat_2d_array = flat_2d_array[STRIP_SIZE:]
            a = a[STRIP_SIZE:]
            n_elements -= STRIP_SIZE
    
    return flat_2d_array