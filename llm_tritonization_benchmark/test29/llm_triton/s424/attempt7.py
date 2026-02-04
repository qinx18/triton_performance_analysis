import triton
import triton.language as tl
import torch

@triton.jit
def s424_kernel(flat_2d_array_ptr, a_ptr, n_elements, vl, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    strip_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = strip_start + offsets
    mask = current_offsets < n_elements
    
    # Read from flat_2d_array[i]
    read_vals = tl.load(flat_2d_array_ptr + current_offsets, mask=mask)
    
    # Read from a[i]
    a_vals = tl.load(a_ptr + current_offsets, mask=mask)
    
    # Compute result
    result = read_vals + a_vals
    
    # Write to xx[i+1] = flat_2d_array[vl + i + 1]
    write_offsets = vl + current_offsets + 1
    tl.store(flat_2d_array_ptr + write_offsets, result, mask=mask)

def s424_triton(a, flat_2d_array, xx):
    vl = 63
    n_elements = a.shape[0] - 1
    
    BLOCK_SIZE = 64
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    s424_kernel[(grid_size,)](
        flat_2d_array,
        a,
        n_elements,
        vl,
        BLOCK_SIZE=BLOCK_SIZE
    )