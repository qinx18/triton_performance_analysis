import triton
import triton.language as tl

@triton.jit
def s122_kernel(a_ptr, b_ptr, n1, n3, N):
    # Calculate the iteration space
    max_iterations = (N - (n1 - 1) + n3 - 1) // n3
    
    # Each program handles one iteration
    pid = tl.program_id(0)
    
    if pid >= max_iterations:
        return
    
    # Calculate i, k for this iteration
    i = n1 - 1 + pid * n3
    k = 1 + pid * 1  # k starts at 1 and increments by j=1 each iteration
    
    # Load and update
    if i < N:
        b_idx = N - k
        if b_idx >= 0 and b_idx < N:
            a_val = tl.load(a_ptr + i)
            b_val = tl.load(b_ptr + b_idx)
            tl.store(a_ptr + i, a_val + b_val)

def s122_triton(a, b):
    N = a.shape[0]
    
    # Calculate parameters from the expected function signature
    # Based on typical TSVC usage patterns
    n1 = 1
    n3 = 2
    
    # Calculate iteration space
    if n1 - 1 >= N or n3 <= 0:
        return
    
    max_iterations = (N - (n1 - 1) + n3 - 1) // n3
    
    if max_iterations <= 0:
        return
    
    # Launch kernel
    grid = (triton.cdiv(max_iterations, 1),)
    s122_kernel[grid](a, b, n1, n3, N)