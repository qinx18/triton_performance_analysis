import triton
import triton.language as tl

@triton.jit
def s111_kernel(a_ptr, b_ptr, N):
    # This kernel must run sequentially due to loop-carried dependency
    # Use single thread to process all iterations
    pid = tl.program_id(0)
    
    if pid == 0:  # Only one thread processes everything
        # Process sequential loop: for i in range(1, N, 2)
        i = 1
        while i < N:
            # Load a[i-1] and b[i]
            a_prev = tl.load(a_ptr + (i - 1))
            b_val = tl.load(b_ptr + i)
            
            # Compute and store a[i] = a[i-1] + b[i]
            result = a_prev + b_val
            tl.store(a_ptr + i, result)
            
            i += 2

def s111_triton(a, b):
    N = a.shape[0]
    
    # Launch single thread since computation must be sequential
    grid = (1,)
    s111_kernel[grid](a, b, N)
    
    return a