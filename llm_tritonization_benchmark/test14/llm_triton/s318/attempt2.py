import torch
import triton
import triton.language as tl

@triton.jit
def s318_kernel(a_ptr, result_ptr, inc, n_elements, BLOCK_SIZE: tl.constexpr):
    # This is a sequential reduction that must be done in a single thread
    block_id = tl.program_id(0)
    if block_id != 0:
        return
    
    # Initialize
    k = 0
    index = 0
    max_val = tl.abs(tl.load(a_ptr))
    k += inc
    
    # Process elements sequentially
    for i in range(1, n_elements):
        if k >= n_elements:
            break
        abs_val = tl.abs(tl.load(a_ptr + k))
        if abs_val > max_val:
            index = i
            max_val = abs_val
        k += inc
    
    # Store results
    tl.store(result_ptr, max_val)
    tl.store(result_ptr + 1, index.to(tl.float32))

def s318_triton(a, inc):
    n_elements = a.shape[0]
    
    # Create result tensor [max_val, index]
    result = torch.zeros(2, dtype=torch.float32, device=a.device)
    
    BLOCK_SIZE = 1024
    grid = (1,)
    
    s318_kernel[grid](
        a, result, inc, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    max_val = result[0].item()
    index = int(result[1].item())
    
    return max_val + index + 1