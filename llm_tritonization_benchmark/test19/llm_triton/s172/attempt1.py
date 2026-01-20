import triton
import triton.language as tl
import torch

@triton.jit
def s172_kernel(a_ptr, b_ptr, n1, n3, N, BLOCK_SIZE: tl.constexpr):
    # Calculate the starting index for this block
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    
    # Create offset vector once
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Calculate how many elements this block can process
    remaining = N - block_start
    block_elements = tl.minimum(BLOCK_SIZE, remaining)
    
    # Process elements in this block
    for local_idx in range(block_elements):
        global_idx = block_start + local_idx
        
        # Calculate the actual array index following the strided pattern
        # We need to find which iteration of the original loop this corresponds to
        # and map it back to the actual array index
        
        # The original loop: for (int i = n1-1; i < N; i += n3)
        # So array_idx = (n1-1) + global_idx * n3
        array_idx = (n1 - 1) + global_idx * n3
        
        # Check bounds
        if array_idx < N:
            # Load values
            a_val = tl.load(a_ptr + array_idx)
            b_val = tl.load(b_ptr + array_idx)
            
            # Compute and store
            result = a_val + b_val
            tl.store(a_ptr + array_idx, result)

def s172_triton(a, b, n1, n3):
    N = a.shape[0]
    
    # Calculate how many elements will be processed by the strided loop
    if n3 <= 0:
        return  # Not vectorizable when n3 == 0
    
    # Count elements that will be processed
    start_idx = n1 - 1
    if start_idx >= N:
        return  # No elements to process
    
    num_elements = (N - start_idx + n3 - 1) // n3
    
    if num_elements <= 0:
        return
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
    
    s172_kernel[grid](a, b, n1, n3, N, BLOCK_SIZE=BLOCK_SIZE)