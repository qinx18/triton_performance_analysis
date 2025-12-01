import torch
import triton
import triton.language as tl

@triton.jit
def s315_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This is a reduction operation to find max value and its index
    # Use single thread since we need to maintain ordering for index finding
    if tl.program_id(0) == 0:
        max_val = tl.load(a_ptr)
        max_idx = 0
        
        for i in range(n_elements):
            val = tl.load(a_ptr + i)
            if val > max_val:
                max_val = val
                max_idx = i
        
        chksum = max_val + max_idx.to(tl.float32)
        
        # Store max_val, max_idx, and chksum
        tl.store(result_ptr, max_val)
        tl.store(result_ptr + 1, max_idx.to(tl.float32))
        tl.store(result_ptr + 2, chksum)

def s315_triton(a):
    n_elements = a.shape[0]
    
    # Result tensor to store [max_val, max_idx, chksum]
    result = torch.zeros(3, dtype=torch.float32, device=a.device)
    
    BLOCK_SIZE = 1024
    grid = (1,)  # Use single block for reduction
    
    s315_kernel[grid](
        a, result, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result