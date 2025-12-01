import torch
import triton
import triton.language as tl

@triton.jit
def s342_kernel(
    a_ptr, b_ptr, output_ptr,
    mask_ptr, prefix_sum_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s342 - conditional assignment with compression.
    Uses prefix sum to determine correct indices for compression.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Boundary check
    mask_boundary = offsets < n_elements
    
    # Load current values and compute condition mask
    a_vals = tl.load(a_ptr + offsets, mask=mask_boundary, other=0.0)
    condition_mask = a_vals > 0.0
    
    # Load precomputed mask and prefix sum
    mask_vals = tl.load(mask_ptr + offsets, mask=mask_boundary, other=0)
    prefix_vals = tl.load(prefix_sum_ptr + offsets, mask=mask_boundary, other=0)
    
    # For elements that satisfy condition, load corresponding b values
    b_indices = prefix_vals - 1  # Convert to 0-based indexing
    b_vals = tl.load(b_ptr + b_indices, mask=(mask_boundary & condition_mask), other=0.0)
    
    # Select between original a values and new b values based on condition
    result = tl.where(condition_mask, b_vals, a_vals)
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask_boundary)

@triton.jit
def prefix_sum_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Simple prefix sum kernel for small arrays.
    For larger arrays, would need a more sophisticated implementation.
    """
    pid = tl.program_id(axis=0)
    
    if pid == 0:
        # Load all elements (assuming small enough to fit in one block)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        vals = tl.load(input_ptr + offsets, mask=mask, other=0)
        
        # Compute prefix sum using scan
        for stride in range(1, BLOCK_SIZE, 1):
            if stride >= n_elements:
                break
            shifted_vals = tl.zeros_like(vals)
            shift_mask = offsets >= stride
            shift_offsets = offsets - stride
            shift_vals = tl.load(input_ptr + shift_offsets, mask=(mask & shift_mask), other=0)
            vals = vals + tl.where(shift_mask, shift_vals, 0)
        
        tl.store(output_ptr + offsets, vals, mask=mask)

def s342_triton(a, b):
    """
    Triton implementation of TSVC s342 - conditional assignment with compression.
    Uses prefix sum to handle compression efficiently on GPU.
    """
    a = a.contiguous()
    b = b.contiguous()
    n_elements = a.shape[0]
    
    # Create mask for condition a > 0
    mask = (a > 0.0).to(torch.int32)
    
    # Compute prefix sum for compression indices
    prefix_sum = torch.zeros_like(mask)
    
    # Use simple approach for prefix sum (for small arrays)
    # For larger arrays, would use a more efficient prefix sum implementation
    BLOCK_SIZE = min(1024, triton.next_power_of_2(n_elements))
    
    if n_elements <= BLOCK_SIZE:
        # Use Triton prefix sum for small arrays
        prefix_sum_kernel[(1,)](
            mask, prefix_sum,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        # Fallback to PyTorch for larger arrays
        prefix_sum = torch.cumsum(mask, dim=0)
    
    # Create output tensor
    output = torch.empty_like(a)
    
    # Launch main kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    s342_kernel[grid](
        a, b, output,
        mask, prefix_sum,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Copy result back to original tensor (to match in-place behavior)
    a.copy_(output)
    return a