import torch
import triton
import triton.language as tl

@triton.jit
def s314_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel finds the maximum value in array a
    pid = tl.program_id(0)
    
    # Load initial value from a[0]
    if pid == 0:
        x = tl.load(a_ptr)
        tl.store(result_ptr, x)
    
    # Wait for initialization
    tl.debug_barrier()
    
    # Each block processes BLOCK_SIZE elements
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values from current block
    values = tl.load(a_ptr + offsets, mask=mask, other=float('-inf'))
    
    # Find local maximum
    local_max = tl.max(values)
    
    # Atomic max to update global result
    tl.atomic_max(result_ptr, local_max)

def s314_triton(a):
    n_elements = a.shape[0]
    
    # Create result tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s314_kernel[grid](
        a, result, n_elements, BLOCK_SIZE
    )
    
    return result.item()