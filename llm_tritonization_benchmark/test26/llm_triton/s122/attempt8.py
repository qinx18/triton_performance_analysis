import triton
import triton.language as tl

@triton.jit
def s122_kernel(a_ptr, b_ptr, n1, n3, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Calculate which iteration this block handles
    iteration = block_start + offsets
    
    # Calculate i values: i = n1-1 + iteration * n3
    i_vals = (n1 - 1) + iteration * n3
    
    # Calculate k values: k = 1 + 2 + ... + (iteration + 1) = (iteration + 1) * (iteration + 2) / 2
    k_vals = (iteration + 1) * (iteration + 2) // 2
    
    # Create mask for valid iterations
    mask = (iteration < ((N - (n1 - 1) + n3 - 1) // n3)) & (i_vals < N)
    
    # Calculate b indices: b[N - k]
    b_indices = N - k_vals
    b_mask = mask & (b_indices >= 0) & (b_indices < N)
    
    # Load values
    a_vals = tl.load(a_ptr + i_vals, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + b_indices, mask=b_mask, other=0.0)
    
    # Update and store
    result = a_vals + b_vals
    tl.store(a_ptr + i_vals, result, mask=mask)

def s122_triton(a, b):
    N = a.shape[0]
    
    # Extract n1 and n3 from typical TSVC patterns
    n1 = 2
    n3 = 3
    
    # Calculate total iterations
    if n1 - 1 >= N or n3 <= 0:
        return
    
    total_iterations = (N - (n1 - 1) + n3 - 1) // n3
    
    if total_iterations <= 0:
        return
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(total_iterations, BLOCK_SIZE),)
    
    s122_kernel[grid](a, b, n1, n3, N, BLOCK_SIZE)