import triton
import triton.language as tl
import torch

@triton.jit
def s172_kernel(a_ptr, b_ptr, n1, n3, N, BLOCK_SIZE: tl.constexpr):
    # Calculate starting position for this program
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process elements in this block
    for i in range(BLOCK_SIZE):
        idx = block_start + i
        
        # Check if we have a valid element in the strided pattern
        if idx >= N:
            break
            
        # Calculate the actual array index based on the strided pattern
        # We need to find which iteration of the original loop this corresponds to
        # Original loop: for (int i = n1-1; i < LEN_1D; i += n3)
        # So we need idx = n1-1 + k*n3 for some k >= 0
        
        # Check if idx can be expressed as n1-1 + k*n3
        if idx >= (n1 - 1):
            offset_from_start = idx - (n1 - 1)
            if offset_from_start % n3 == 0:
                # This is a valid index in the strided pattern
                mask = idx < N
                if mask:
                    a_val = tl.load(a_ptr + idx)
                    b_val = tl.load(b_ptr + idx)
                    result = a_val + b_val
                    tl.store(a_ptr + idx, result)

def s172_triton(a, b, n1, n3):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate total number of elements that will be processed
    # This is the count of valid indices in the strided pattern
    if n1 <= N and n3 > 0:
        num_elements = ((N - (n1 - 1) - 1) // n3) + 1
        num_elements = max(0, min(num_elements, N))
        grid_size = triton.cdiv(N, BLOCK_SIZE)
    else:
        grid_size = 1
    
    grid = (grid_size,)
    
    s172_kernel[grid](
        a, b, n1, n3, N,
        BLOCK_SIZE=BLOCK_SIZE
    )