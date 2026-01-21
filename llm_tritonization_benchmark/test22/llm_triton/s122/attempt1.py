import triton
import triton.language as tl
import torch

@triton.jit
def s122_kernel(a_ptr, b_ptr, n1, n3, N, BLOCK_SIZE: tl.constexpr):
    # Calculate the number of iterations in the original loop
    num_iterations = (N - (n1 - 1) + n3 - 1) // n3
    
    # Get program ID and calculate block range
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector for vectorized operations
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process iterations in blocks
    for block_offset in range(0, BLOCK_SIZE):
        iter_idx = block_start + block_offset
        
        # Check if this iteration is valid
        if iter_idx >= num_iterations:
            break
            
        # Calculate i, j, k for this iteration
        i = (n1 - 1) + iter_idx * n3
        j = 1
        k = j * (iter_idx + 1)  # k accumulates j for each iteration
        
        # Check bounds
        if i < N:
            b_idx = N - k
            if b_idx >= 0 and b_idx < N:
                # Load values
                a_val = tl.load(a_ptr + i)
                b_val = tl.load(b_ptr + b_idx)
                
                # Update a[i]
                tl.store(a_ptr + i, a_val + b_val)

def s122_triton(a, b, n1, n3):
    N = a.shape[0]
    
    # Calculate the number of loop iterations
    if n1 - 1 >= N:
        return  # No iterations
    
    num_iterations = (N - (n1 - 1) + n3 - 1) // n3
    if num_iterations <= 0:
        return
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(num_iterations, BLOCK_SIZE),)
    
    # Launch kernel
    s122_kernel[grid](
        a, b, n1, n3, N,
        BLOCK_SIZE=BLOCK_SIZE
    )