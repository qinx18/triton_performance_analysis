import torch
import triton
import triton.language as tl

@triton.jit
def s318_kernel(a_ptr, inc, n, output_ptr):
    # This is a sequential reduction that needs to be done in a single thread
    # to maintain the exact semantics of the original C code
    
    if tl.program_id(0) != 0:
        return
    
    k = 0
    index = 0
    
    # Load first element and compute its absolute value
    first_val = tl.load(a_ptr + k)
    max_val = tl.abs(first_val)
    k += inc
    
    # Sequential loop to find max absolute value and its index
    i = 1
    while i < n:
        # Bounds check - early exit if k would be out of bounds
        valid = k < n * inc
        if not valid:
            i = n  # Force loop exit
        else:
            current_val = tl.load(a_ptr + k)
            abs_val = tl.abs(current_val)
            
            # Update max and index if we found a larger absolute value
            # Note: original C code uses goto to skip update if abs_val <= max
            if abs_val > max_val:
                index = i
                max_val = abs_val
                
            k += inc
            i += 1
    
    # Store the result (max + index + 1 as per C code return)
    result = max_val + tl.cast(index, tl.float32) + 1.0
    tl.store(output_ptr, result)

def s318_triton(a, inc):
    n = a.shape[0]
    
    # Output tensor to store the result
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single block/thread to maintain sequential semantics
    grid = (1,)
    s318_kernel[grid](
        a, inc, n, output
    )
    
    return output.item()