import torch
import triton
import triton.language as tl

@triton.jit
def s318_kernel(a_ptr, result_ptr, inc, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel finds the maximum absolute value and its index with stride inc
    # Each block processes independently and finds local max, then we reduce
    
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Initialize with first valid element (index 0)
    local_max = tl.abs(tl.load(a_ptr, mask=block_start == 0, other=0.0))
    local_index = 0
    
    # Process elements starting from index 1 with stride inc
    for i in range(1, n_elements):
        k = i * inc
        if k >= n_elements:
            break
            
        # Load current element
        current_val = tl.abs(tl.load(a_ptr + k, mask=k < n_elements, other=0.0))
        
        # Update max and index if current value is greater
        update_condition = current_val > local_max
        local_max = tl.where(update_condition, current_val, local_max)
        local_index = tl.where(update_condition, i, local_index)
    
    # Store result (max + index as checksum)
    if block_id == 0:  # Only first block writes result
        chksum = local_max + local_index
        tl.store(result_ptr, chksum)
        tl.store(result_ptr + 1, local_max)
        tl.store(result_ptr + 2, local_index)

@triton.jit 
def s318_reduction_kernel(a_ptr, result_ptr, inc, n_elements):
    # Sequential implementation matching C code exactly
    k = 0
    index = 0
    max_val = tl.abs(tl.load(a_ptr))
    k += inc
    
    for i in range(1, n_elements):
        if k >= n_elements:
            k += inc
            continue
            
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
    s318_reduction_kernel[(1,)](
        a, result, inc, n_elements
    )
    
    max_val = result[1].item()
    index = result[2].item()
    chksum = result[0].item()
    
    return max_val, index, chksum