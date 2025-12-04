import triton
import triton.language as tl
import torch

@triton.jit
def s122_kernel(a_ptr, b_ptr, n1, n3, LEN_1D, BLOCK_SIZE: tl.constexpr):
    # Calculate the number of iterations in the original loop
    num_iterations = (LEN_1D - (n1 - 1) + n3 - 1) // n3
    
    # Process iterations in blocks
    for block_start in range(0, num_iterations, BLOCK_SIZE):
        # Calculate which iteration indices this block handles
        iteration_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        iteration_mask = iteration_offsets < num_iterations
        
        # Calculate the corresponding i values: i = n1-1 + iteration * n3
        i_values = (n1 - 1) + iteration_offsets * n3
        i_mask = iteration_mask & (i_values < LEN_1D)
        
        # Calculate k values: k = j * iteration = 1 * iteration = iteration
        k_values = iteration_offsets + 1  # k = j + j*iteration where j=1
        
        # Calculate b indices: LEN_1D - k
        b_indices = LEN_1D - k_values
        b_mask = i_mask & (b_indices >= 0) & (b_indices < LEN_1D)
        
        # Load values
        a_vals = tl.load(a_ptr + i_values, mask=b_mask, other=0.0)
        b_vals = tl.load(b_ptr + b_indices, mask=b_mask, other=0.0)
        
        # Compute and store
        result = a_vals + b_vals
        tl.store(a_ptr + i_values, result, mask=b_mask)

def s122_triton(a, b, n1, n3):
    LEN_1D = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch kernel
    s122_kernel[(1,)](
        a, b, n1, n3, LEN_1D, BLOCK_SIZE
    )