import torch
import triton
import triton.language as tl

@triton.jit
def s315_kernel(a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Initialize reduction values
    max_val = tl.load(a_ptr)  # a[0]
    max_idx = 0
    
    # Pre-define offsets once at kernel start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process array in blocks to find maximum value and index
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load block of values
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
        
        # Find maximum in this block
        block_max = tl.max(vals, axis=0)
        
        # Update global maximum if block maximum is larger
        if block_max > max_val:
            max_val = block_max
            # Find the index of maximum in this block
            for i in range(BLOCK_SIZE):
                if block_start + i < n_elements:
                    val = tl.load(a_ptr + block_start + i)
                    if val > max_val or (val == max_val and val == block_max and block_start + i > max_idx):
                        if val == block_max:
                            max_val = val
                            max_idx = block_start + i
    
    # Store results (using first element of array as temporary storage)
    tl.store(a_ptr + n_elements, max_val)  # Store max value
    tl.store(a_ptr + n_elements + 1, max_idx)  # Store max index

@triton.jit
def s315_kernel_simple(a_ptr, result_ptr, n_elements):
    # Sequential search for maximum (since this is inherently sequential)
    max_val = tl.load(a_ptr)  # a[0] 
    max_idx = 0
    
    # Sequential scan through array
    for i in range(n_elements):
        val = tl.load(a_ptr + i)
        if val > max_val:
            max_val = val
            max_idx = i
    
    # Store results
    tl.store(result_ptr, max_val)
    tl.store(result_ptr + 1, max_idx)

def s315_triton(a):
    n_elements = a.shape[0]
    
    # Create result tensor to store max_val and max_idx
    result = torch.zeros(2, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single thread since this is a sequential reduction
    s315_kernel_simple[(1,)](
        a, result, n_elements
    )
    
    max_val = result[0].item()
    max_idx = int(result[1].item())
    chksum = max_val + max_idx
    
    return max_val, max_idx, chksum