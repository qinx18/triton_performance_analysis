import torch
import triton
import triton.language as tl

@triton.jit
def s424_kernel(flat_2d_array_ptr, a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Calculate number of strips
    num_strips = (n_elements + 63) // 64
    
    for strip_id in range(num_strips):
        strip_start = strip_id * 64
        strip_end = min(strip_start + 64, n_elements)
        
        # Process elements in this strip
        idx = strip_start + offsets
        mask = (offsets < (strip_end - strip_start)) & (idx < n_elements)
        
        # Read from flat_2d_array[i] and a[i]
        flat_vals = tl.load(flat_2d_array_ptr + idx, mask=mask)
        a_vals = tl.load(a_ptr + idx, mask=mask)
        
        # Compute result
        result = flat_vals + a_vals
        
        # Write to xx[i+1] which is flat_2d_array[i+64]
        write_idx = idx + 64
        write_mask = mask & (write_idx < (flat_2d_array_ptr.shape[0] if hasattr(flat_2d_array_ptr, 'shape') else n_elements + 64))
        tl.store(flat_2d_array_ptr + write_idx, result, mask=write_mask)

def s424_triton(a, flat_2d_array, xx):
    n_elements = a.shape[0] - 1  # LEN_1D - 1
    
    # Use block size <= 64 to respect strip constraints
    BLOCK_SIZE = 64
    
    # Launch single block since we process strips sequentially inside kernel
    s424_kernel[(1,)](
        flat_2d_array,
        a,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )