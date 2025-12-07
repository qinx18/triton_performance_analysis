import torch
import triton
import triton.language as tl

@triton.jit
def s172_kernel(a_ptr, b_ptr, n1, n3, LEN_1D, BLOCK_SIZE: tl.constexpr):
    # Calculate number of elements to process
    start_idx = n1 - 1
    num_elements = (LEN_1D - start_idx + n3 - 1) // n3
    
    # Get block index
    block_idx = tl.program_id(0)
    
    # Calculate element index in the strided sequence
    element_idx = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid elements in this block
    mask = element_idx < num_elements
    
    # Calculate actual array indices
    array_indices = start_idx + element_idx * n3
    
    # Load values
    a_vals = tl.load(a_ptr + array_indices, mask=mask)
    b_vals = tl.load(b_ptr + array_indices, mask=mask)
    
    # Compute a[i] += b[i]
    result = a_vals + b_vals
    
    # Store result
    tl.store(a_ptr + array_indices, result, mask=mask)

def s172_triton(a, b, n1, n3):
    LEN_1D = a.shape[0]
    
    # Calculate number of elements to process
    start_idx = n1 - 1
    if start_idx >= LEN_1D or n3 <= 0:
        return
    
    num_elements = (LEN_1D - start_idx + n3 - 1) // n3
    
    if num_elements <= 0:
        return
    
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(num_elements, BLOCK_SIZE)
    
    s172_kernel[(num_blocks,)](
        a, b, n1, n3, LEN_1D,
        BLOCK_SIZE=BLOCK_SIZE
    )