import torch
import triton
import triton.language as tl

@triton.jit
def s314_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This is a reduction kernel to find maximum value
    # Each block processes BLOCK_SIZE elements and finds local max
    
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    mask = current_offsets < n_elements
    
    # Load block of data
    vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
    
    # Find maximum in this block
    block_max = tl.max(vals)
    
    # Store block result
    tl.store(result_ptr + pid, block_max)

def s314_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate number of blocks needed
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create output tensor for block results
    block_results = torch.empty(n_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    s314_kernel[(n_blocks,)](
        a, block_results, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Reduce block results on CPU to find global maximum
    # Start with a[0] as in original code
    x = a[0].item()
    block_max = torch.max(block_results).item()
    
    return max(x, block_max)