import torch
import triton
import triton.language as tl

@triton.jit
def s122_kernel(a_ptr, b_ptr, n1, n3, LEN_1D, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel for s122 computation
    Each thread block processes multiple iterations of the original loop
    """
    # Get program ID and calculate starting iteration
    pid = tl.program_id(0)
    
    # Calculate the starting index for this block
    block_start = pid * BLOCK_SIZE
    
    # Generate offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Calculate actual indices in the original loop: i = n1-1 + iteration*n3
    indices = (n1 - 1) + offsets * n3
    
    # Mask for valid indices within bounds
    mask = (indices < LEN_1D) & (offsets >= 0)
    
    # Calculate k values: k = j * (iteration + 1), where j=1
    k_values = offsets + 1
    
    # Calculate b indices: LEN_1D - k
    b_indices = LEN_1D - k_values
    
    # Load values from a and b with masking
    a_vals = tl.load(a_ptr + indices, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + b_indices, mask=mask & (b_indices >= 0) & (b_indices < LEN_1D), other=0.0)
    
    # Perform computation: a[i] += b[LEN_1D - k]
    result = a_vals + b_vals
    
    # Store result back to a
    tl.store(a_ptr + indices, result, mask=mask)

def s122_triton(a, b, n1, n3):
    """
    Triton implementation of TSVC s122
    Optimizes the sequential loop by parallelizing iterations
    """
    a = a.contiguous()
    b = b.contiguous()
    
    LEN_1D = len(a)
    
    # Calculate total number of iterations in the original loop
    if n1 - 1 >= LEN_1D or n3 <= 0:
        return a
    
    num_iterations = (LEN_1D - (n1 - 1) + n3 - 1) // n3
    
    if num_iterations <= 0:
        return a
    
    # Choose block size based on number of iterations
    BLOCK_SIZE = min(1024, triton.next_power_of_2(num_iterations))
    
    # Calculate grid size
    grid = (triton.cdiv(num_iterations, BLOCK_SIZE),)
    
    # Launch kernel
    s122_kernel[grid](
        a, b, n1, n3, LEN_1D, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a