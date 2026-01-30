import torch
import triton
import triton.language as tl

@triton.jit
def s122_kernel(a_ptr, b_ptr, n1, n3, N, BLOCK_SIZE: tl.constexpr):
    # Calculate the number of iterations in the loop
    num_iters = (N - (n1 - 1) + n3 - 1) // n3
    
    # Get block ID and calculate which iterations this block handles
    block_id = tl.program_id(0)
    
    # Calculate how many iterations per block
    iters_per_block = triton.cdiv(num_iters, tl.num_programs(0))
    
    # Calculate start and end iteration for this block
    iter_start = block_id * iters_per_block
    iter_end = tl.minimum(iter_start + iters_per_block, num_iters)
    
    # Process iterations assigned to this block
    for iter_idx in range(iter_start, iter_end):
        # Calculate i and k for this iteration
        i = (n1 - 1) + iter_idx * n3
        k = 1 + iter_idx  # j=1, so k = j * (iter_idx + 1) = iter_idx + 1
        
        # Check bounds
        if i < N:
            b_idx = N - k
            if b_idx >= 0:
                # Load values
                a_val = tl.load(a_ptr + i)
                b_val = tl.load(b_ptr + b_idx)
                
                # Update a[i]
                tl.store(a_ptr + i, a_val + b_val)

def s122_triton(a, b, n1, n3):
    N = a.shape[0]
    
    # Calculate number of iterations
    if n1 - 1 >= N or n3 <= 0:
        return
    
    num_iters = (N - (n1 - 1) + n3 - 1) // n3
    if num_iters <= 0:
        return
    
    # Launch kernel with appropriate grid size
    BLOCK_SIZE = 256
    grid_size = min(triton.cdiv(num_iters, 4), 1024)  # At least 4 iterations per block
    
    s122_kernel[(grid_size,)](
        a, b, n1, n3, N, BLOCK_SIZE
    )