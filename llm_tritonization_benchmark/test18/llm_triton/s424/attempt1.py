import torch
import triton
import triton.language as tl

@triton.jit
def s424_kernel(flat_2d_array_ptr, a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Each block processes one strip sequentially
    strip_id = tl.program_id(0)
    strip_start = strip_id * 64
    idx = strip_start + offsets
    
    # Mask for valid indices
    mask = idx < n_elements
    
    # Load from read position (idx + 0)
    flat_vals = tl.load(flat_2d_array_ptr + idx, mask=mask, other=0.0)
    a_vals = tl.load(a_ptr + idx, mask=mask, other=0.0)
    
    # Compute result
    result = flat_vals + a_vals
    
    # Store to write position (idx + 64)
    write_mask = mask & ((idx + 64) < (n_elements + 64))
    tl.store(flat_2d_array_ptr + idx + 64, result, mask=write_mask)

def s424_triton(a, flat_2d_array):
    n_elements = a.shape[0] - 1  # LEN_1D - 1
    
    # Strip size is 64 (due to dependency analysis)
    STRIP_SIZE = 64
    BLOCK_SIZE = min(STRIP_SIZE, 64)  # Must be <= 64
    
    # Calculate number of strips
    num_strips = triton.cdiv(n_elements, STRIP_SIZE)
    
    # Process strips sequentially (due to loop-carried dependency)
    for strip_id in range(num_strips):
        strip_start = strip_id * STRIP_SIZE
        remaining = min(STRIP_SIZE, n_elements - strip_start)
        
        if remaining > 0:
            # Launch kernel for this strip only
            s424_kernel[(1,)](
                flat_2d_array,
                a,
                strip_start + remaining,
                BLOCK_SIZE=min(BLOCK_SIZE, triton.next_power_of_2(remaining))
            )