import torch
import triton
import triton.language as tl

@triton.jit
def s322_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel processes the recurrence sequentially since each element depends on previous ones
    # We'll process one element per program
    pid = tl.program_id(axis=0)
    
    # Each program handles one iteration of the sequential loop
    i = pid + 2
    
    if i < n_elements:
        # Load the required values
        a_i = tl.load(a_ptr + i)
        a_i_minus_1 = tl.load(a_ptr + i - 1)
        a_i_minus_2 = tl.load(a_ptr + i - 2)
        b_i = tl.load(b_ptr + i)
        c_i = tl.load(c_ptr + i)
        
        # Compute: a[i] = a[i] + a[i-1] * b[i] + a[i-2] * c[i]
        result = a_i + a_i_minus_1 * b_i + a_i_minus_2 * c_i
        
        # Store result
        tl.store(a_ptr + i, result)

def s322_triton(a, b, c):
    n_elements = a.shape[0]
    
    # Since this is a second-order linear recurrence, we need to process sequentially
    # We'll launch kernels iteratively to maintain the dependency chain
    BLOCK_SIZE = 128
    
    # Process the recurrence sequentially, one element at a time
    for i in range(2, n_elements):
        grid = (1,)
        s322_kernel[grid](
            a, b, c, n_elements, BLOCK_SIZE
        )
        # Synchronize to ensure the write is complete before next iteration
        torch.cuda.synchronize()
    
    return a