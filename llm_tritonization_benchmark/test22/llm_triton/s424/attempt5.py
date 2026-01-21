import torch
import triton
import triton.language as tl

@triton.jit
def s424_kernel(flat_2d_array_ptr, a_ptr, strip_start, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets
    
    # Mask for valid elements in this strip
    mask = idx < n_elements
    
    # Read from flat_2d_array[idx] and a[idx]
    flat_vals = tl.load(flat_2d_array_ptr + idx, mask=mask)
    a_vals = tl.load(a_ptr + idx, mask=mask)
    
    # Compute result
    result = flat_vals + a_vals
    
    # Write to xx[idx+1] which is flat_2d_array[idx+64]
    write_idx = idx + 64
    # xx[i+1] means we write to position i+1, which corresponds to flat_2d_array[idx+1+63] = flat_2d_array[idx+64]
    # But actually xx[i+1] where xx = flat_2d_array + 63 means flat_2d_array[i+1+63] = flat_2d_array[i+64]
    # So we write to flat_2d_array at offset idx + 1 + 63 = idx + 64
    total_elements = n_elements + 64
    write_mask = mask & (write_idx < total_elements)
    tl.store(flat_2d_array_ptr + write_idx, result, mask=write_mask)

def s424_triton(a, flat_2d_array, xx):
    n_elements = a.shape[0] - 1  # LEN_1D - 1
    STRIP_SIZE = 64
    BLOCK_SIZE = 64
    
    # Process in strips sequentially due to loop-carried dependency
    for strip_start in range(0, n_elements, STRIP_SIZE):
        # Process current strip
        s424_kernel[(1,)](
            flat_2d_array,
            a,
            strip_start,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )