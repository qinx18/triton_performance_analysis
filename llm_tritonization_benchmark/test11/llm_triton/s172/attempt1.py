import torch
import triton
import triton.language as tl

@triton.jit
def s172_kernel(a_ptr, b_ptr, n1, n3, len_1d, BLOCK_SIZE: tl.constexpr):
    # Calculate total number of elements in the strided sequence
    start_idx = n1 - 1
    if start_idx >= len_1d or n3 <= 0:
        return
    
    # Calculate number of valid indices in the sequence
    num_elements = (len_1d - start_idx + n3 - 1) // n3
    
    # Get block start index
    block_start = tl.program_id(0) * BLOCK_SIZE
    
    # Create offset vector
    offsets = tl.arange(0, BLOCK_SIZE)
    block_offsets = block_start + offsets
    
    # Mask for valid elements in this block
    mask = block_offsets < num_elements
    
    # Convert block indices to actual array indices using stride
    array_indices = start_idx + block_offsets * n3
    
    # Load values with masking
    a_vals = tl.load(a_ptr + array_indices, mask=mask)
    b_vals = tl.load(b_ptr + array_indices, mask=mask)
    
    # Compute a[i] += b[i]
    result = a_vals + b_vals
    
    # Store result
    tl.store(a_ptr + array_indices, result, mask=mask)

def s172_triton(a, b, n1, n3):
    len_1d = a.shape[0]
    
    # Calculate total number of elements in the strided sequence
    start_idx = n1 - 1
    if start_idx >= len_1d or n3 <= 0:
        return
    
    num_elements = (len_1d - start_idx + n3 - 1) // n3
    
    if num_elements <= 0:
        return
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
    
    s172_kernel[grid](
        a, b, n1, n3, len_1d,
        BLOCK_SIZE=BLOCK_SIZE
    )