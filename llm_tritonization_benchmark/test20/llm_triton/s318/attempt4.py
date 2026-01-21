import triton
import triton.language as tl
import torch

@triton.jit
def s318_kernel(a_ptr, output_ptr, inc, n, BLOCK_SIZE: tl.constexpr):
    # This is a sequential reduction that needs to be done in one thread
    # Use only the first thread of the first block
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Initialize with first element
    k = 0
    index = 0
    max_val = tl.load(a_ptr + k)
    max_val = tl.where(max_val >= 0, max_val, -max_val)  # abs
    k += inc
    
    # Sequential search for maximum absolute value
    for i in range(1, n):
        # Check bounds
        valid = k < n * inc
        if valid:
            current_val = tl.load(a_ptr + k)
            current_abs = tl.where(current_val >= 0, current_val, -current_val)  # abs
            if current_abs > max_val:
                index = i
                max_val = current_abs
        k += inc
    
    # Store result: max + index + 1 (as per return statement)
    result = max_val + index + 1
    tl.store(output_ptr, result)

def s318_triton(a, abs, inc):
    n = a.shape[0]
    
    # Output tensor for the result
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Use single thread since this is inherently sequential
    BLOCK_SIZE = 1
    grid = (1,)
    
    s318_kernel[grid](
        a, output, inc, n,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output.item()