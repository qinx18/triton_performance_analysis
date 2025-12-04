import torch
import triton
import triton.language as tl

@triton.jit 
def s318_kernel(a_ptr, result_ptr, inc, n_elements):
    # Sequential implementation matching C code exactly
    k = 0
    index = 0
    max_val = tl.abs(tl.load(a_ptr))
    k += inc
    
    for i in range(1, n_elements):
        if k >= n_elements:
            k += inc
        else:
            current_abs = tl.abs(tl.load(a_ptr + k))
            
            # if (ABS(a[k]) <= max) goto L5; else update max and index
            update_needed = current_abs > max_val
            max_val = tl.where(update_needed, current_abs, max_val)
            index = tl.where(update_needed, i, index)
            
            k += inc
    
    chksum = max_val + index
    tl.store(result_ptr, chksum)
    tl.store(result_ptr + 1, max_val)
    tl.store(result_ptr + 2, index)

def s318_triton(a, inc):
    n_elements = a.shape[0]
    
    # Result tensor: [chksum, max_val, index]
    result = torch.zeros(3, device=a.device, dtype=a.dtype)
    
    # Use sequential kernel for exact C code matching
    s318_kernel[(1,)](
        a, result, inc, n_elements
    )
    
    max_val = result[1].item()
    index = result[2].item()
    chksum = result[0].item()
    
    return max_val, index, chksum