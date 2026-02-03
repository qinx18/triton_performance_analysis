import torch
import triton
import triton.language as tl

@triton.jit
def s321_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel implements a first-order linear recurrence
    # a[i] += a[i-1] * b[i] for i from 1 to n_elements-1
    # Due to the dependency, we must process elements sequentially
    
    # Each program handles one block sequentially
    program_id = tl.program_id(0)
    block_start = program_id * BLOCK_SIZE
    
    # Process elements sequentially within this block
    for local_i in range(BLOCK_SIZE):
        global_i = block_start + local_i + 1  # Start from 1, not 0
        
        if global_i < n_elements:
            # Load a[i-1], a[i], and b[i]
            a_prev = tl.load(a_ptr + global_i - 1)
            a_curr = tl.load(a_ptr + global_i)
            b_curr = tl.load(b_ptr + global_i)
            
            # Compute a[i] += a[i-1] * b[i]
            new_val = a_curr + a_prev * b_curr
            
            # Store the result
            tl.store(a_ptr + global_i, new_val)

def s321_triton(a, b):
    n_elements = a.shape[0]
    
    # For recurrence relations, we need smaller block sizes to maintain dependencies
    BLOCK_SIZE = 32
    
    # We need to process n_elements-1 elements (starting from index 1)
    elements_to_process = n_elements - 1
    num_programs = triton.cdiv(elements_to_process, BLOCK_SIZE)
    
    # Launch the kernel
    grid = (num_programs,)
    s321_kernel[grid](
        a, b, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )