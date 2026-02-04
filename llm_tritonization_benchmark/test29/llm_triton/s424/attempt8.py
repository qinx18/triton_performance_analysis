import triton
import triton.language as tl
import torch

@triton.jit
def s424_kernel(flat_2d_array_ptr, a_ptr, n_elements, vl, STRIP_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for strip_start in range(0, n_elements, STRIP_SIZE):
        current_offsets = strip_start + offsets
        mask = current_offsets < n_elements
        
        # Read from flat_2d_array[i] and a[i]
        flat_vals = tl.load(flat_2d_array_ptr + current_offsets, mask=mask)
        a_vals = tl.load(a_ptr + current_offsets, mask=mask)
        
        # Compute result
        result = flat_vals + a_vals
        
        # Write to xx[i+1] = flat_2d_array[vl + i + 1]
        write_offsets = vl + current_offsets + 1
        write_mask = current_offsets < n_elements
        tl.store(flat_2d_array_ptr + write_offsets, result, mask=write_mask)

def s424_triton(a, flat_2d_array, xx):
    vl = 63
    n_elements = a.shape[0] - 1
    
    STRIP_SIZE = 64
    BLOCK_SIZE = 64
    
    s424_kernel[(1,)](
        flat_2d_array,
        a,
        n_elements,
        vl,
        STRIP_SIZE=STRIP_SIZE,
        BLOCK_SIZE=BLOCK_SIZE
    )