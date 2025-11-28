import torch
import triton
import triton.language as tl

@triton.jit
def s318_kernel(a_ptr, inc, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel finds the maximum absolute value and its index
    # Each block processes the entire array to find local max
    block_id = tl.program_id(0)
    
    if block_id > 0:
        return
    
    # Initialize with first element
    k = 0
    max_val = tl.abs(tl.load(a_ptr + k))
    max_idx = 0
    k += inc
    
    # Process remaining elements
    for i in range(1, n_elements):
        if k < n_elements:
            val = tl.abs(tl.load(a_ptr + k))
            # Update max and index if current value is greater
            if val > max_val:
                max_val = val
                max_idx = i
        k += inc
    
    # Store results (max_val and max_idx) - we'll retrieve them on CPU
    # For simplicity, store at the beginning of array a (will be restored)
    tl.store(a_ptr, max_val)
    tl.store(a_ptr + 1, max_idx)

@triton.jit
def s318_reduction_kernel(a_ptr, inc, n_elements, 
                         max_vals_ptr, max_idxs_ptr,
                         BLOCK_SIZE: tl.constexpr):
    # Each thread block finds local maximum
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Convert linear offsets to k indices using increment
    k_offsets = offsets * inc
    mask = (offsets < n_elements) & (k_offsets < n_elements)
    
    # Load values with masking
    vals = tl.load(a_ptr + k_offsets, mask=mask, other=0.0)
    abs_vals = tl.abs(vals)
    
    # Find local maximum
    local_max = tl.max(abs_vals, axis=0)
    
    # Find index of maximum value
    max_mask = abs_vals == local_max
    indices = tl.arange(0, BLOCK_SIZE)
    valid_indices = tl.where(max_mask & mask, offsets, n_elements)
    local_idx = tl.min(valid_indices, axis=0)
    
    # Store block results
    if pid < tl.cdiv(n_elements, BLOCK_SIZE):
        tl.store(max_vals_ptr + pid, local_max)
        tl.store(max_idxs_ptr + pid, local_idx)

def s318_triton(a, inc=1):
    n_elements = a.shape[0]
    
    # Simple approach: single thread processes entire array sequentially
    # to match the original algorithm exactly
    BLOCK_SIZE = 1024
    
    # Save original values
    orig_val0 = a[0].clone()
    orig_val1 = a[1].clone() if n_elements > 1 else torch.tensor(0.0)
    
    # Launch kernel
    grid = (1,)
    s318_kernel[grid](
        a, inc, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Retrieve results
    max_val = a[0].item()
    max_idx = int(a[1].item())
    
    # Restore original values
    a[0] = orig_val0
    if n_elements > 1:
        a[1] = orig_val1
    
    # Calculate checksum as in original
    chksum = max_val + float(max_idx)
    
    return chksum