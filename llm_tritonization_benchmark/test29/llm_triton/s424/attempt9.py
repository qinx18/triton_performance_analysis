import triton
import triton.language as tl
import torch

@triton.jit
def s424_kernel(flat_2d_array_ptr, a_ptr, n_elements, vl, STRIP_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    strip_id = pid // triton.cdiv(STRIP_SIZE, BLOCK_SIZE)
    block_id_in_strip = pid % triton.cdiv(STRIP_SIZE, BLOCK_SIZE)
    
    strip_start = strip_id * STRIP_SIZE
    block_start = strip_start + block_id_in_strip * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Read from flat_2d_array[i] and a[i]
    flat_vals = tl.load(flat_2d_array_ptr + current_offsets, mask=mask)
    a_vals = tl.load(a_ptr + current_offsets, mask=mask)
    
    # Compute result
    result = flat_vals + a_vals
    
    # Write to xx[i+1] = flat_2d_array[vl + i + 1]
    write_offsets = vl + current_offsets + 1
    tl.store(flat_2d_array_ptr + write_offsets, result, mask=mask)

def s424_triton(a, flat_2d_array, xx):
    vl = 63
    n_elements = a.shape[0] - 1
    
    STRIP_SIZE = 64
    BLOCK_SIZE = 64
    
    num_strips = triton.cdiv(n_elements, STRIP_SIZE)
    blocks_per_strip = triton.cdiv(STRIP_SIZE, BLOCK_SIZE)
    grid_size = num_strips * blocks_per_strip
    
    s424_kernel[(grid_size,)](
        flat_2d_array,
        a,
        n_elements,
        vl,
        STRIP_SIZE=STRIP_SIZE,
        BLOCK_SIZE=BLOCK_SIZE
    )