import triton
import triton.language as tl
import torch

@triton.jit
def s112_kernel(
    a_ptr,
    a_copy_ptr,
    b_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Read indices (i values in reverse order)
    read_offsets = n_elements - 1 - offsets
    read_mask = read_offsets >= 0
    
    # Write indices (i+1 values in reverse order)  
    write_offsets = read_offsets + 1
    write_mask = (write_offsets < n_elements) & read_mask
    
    # Load from read-only copy and b array
    a_vals = tl.load(a_copy_ptr + read_offsets, mask=read_mask, other=0.0)
    b_vals = tl.load(b_ptr + read_offsets, mask=read_mask, other=0.0)
    
    # Compute result
    result = a_vals + b_vals
    
    # Store to original array
    tl.store(a_ptr + write_offsets, result, mask=write_mask)

def s112_triton(a, b):
    n_elements = len(a) - 1  # We process indices 0 to LEN_1D-2
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s112_kernel[grid](
        a,
        a_copy,
        b,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )