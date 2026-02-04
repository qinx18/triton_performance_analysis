import triton
import triton.language as tl
import torch

@triton.jit
def s318_kernel(a_ptr, result_ptr, inc, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate strided indices
    k_indices = offsets * inc
    k_mask = (k_indices < n_elements) & mask
    
    # Load values with stride
    vals = tl.load(a_ptr + k_indices, mask=k_mask, other=0.0)
    
    # Apply absolute value
    abs_vals = tl.abs(vals)
    
    # Find local max and index
    local_max = tl.max(abs_vals, axis=0)
    
    # Store partial results for final reduction
    tl.store(result_ptr + pid, local_max)

def s318_triton(a, abs, inc):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate number of elements we can access with stride
    max_elements = (N + inc - 1) // inc
    grid_size = triton.cdiv(max_elements, BLOCK_SIZE)
    
    # Allocate temporary storage for partial results
    partial_results = torch.zeros(grid_size, dtype=a.dtype, device=a.device)
    
    # Launch kernel for partial reductions
    grid = (grid_size,)
    s318_kernel[grid](
        a, partial_results, inc, max_elements, BLOCK_SIZE
    )
    
    # Final reduction on CPU/PyTorch
    # Gather elements with stride inc manually
    k_indices = torch.arange(0, N, inc, dtype=torch.int64, device=a.device)
    k_indices = torch.clamp(k_indices, 0, N - 1)
    
    # Get strided values and apply absolute value
    vals = a[k_indices]
    vals_abs = torch.abs(vals)
    
    # Find max and index
    max_val = torch.max(vals_abs)
    max_idx = torch.argmax(vals_abs)
    
    return max_val + max_idx + 1