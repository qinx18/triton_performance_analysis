import torch
import triton
import triton.language as tl

@triton.jit
def s318_kernel(a_ptr, n, inc, result_ptr, BLOCK_SIZE: tl.constexpr):
    # Use single thread for sequential strided access
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Initialize
    k = 0
    index = 0
    
    # Load first element
    first_val = tl.load(a_ptr + 0)
    max_val = tl.abs(first_val)
    k = k + inc
    
    # Sequential processing for strided access
    for i in range(1, n):
        # Check bounds
        if k < n:
            current_val = tl.load(a_ptr + k)
            abs_val = tl.abs(current_val)
            
            # Update max and index if current is greater (note: C code uses <= for goto)
            if abs_val > max_val:
                index = i
                max_val = abs_val
        
        k = k + inc
    
    # Store results - ensure both are float32
    tl.store(result_ptr + 0, max_val)
    tl.store(result_ptr + 1, tl.cast(index, tl.float32))

def s318_triton(a, inc):
    n = a.shape[0]
    
    # Create result tensor to store max_val and index
    result = torch.zeros(2, dtype=torch.float32, device=a.device)
    
    grid = (1,)  # Single thread for sequential reduction
    BLOCK_SIZE = 256
    
    s318_kernel[grid](
        a, n, inc, result,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    max_val = result[0].item()
    index = int(result[1].item())
    
    # Return exact value as C code: max + index + 1
    return max_val + float(index) + 1.0