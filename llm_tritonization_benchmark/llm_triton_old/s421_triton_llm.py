import torch
import triton
import triton.language as tl

@triton.jit
def s421_kernel(
    a_ptr, xx_ptr, yy_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s421: xx[i] = yy[i+1] + a[i] where yy = xx initially
    Key optimizations:
    - Coalesced memory access with BLOCK_SIZE chunks
    - Single kernel launch instead of element-wise operations
    - Efficient masking for boundary conditions
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid indices (we process n_elements-1 elements)
    mask = offsets < (n_elements - 1)
    
    # Load a[i] and yy[i+1] with masking
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    yy_vals = tl.load(yy_ptr + offsets + 1, mask=mask, other=0.0)
    
    # Compute xx[i] = yy[i+1] + a[i]
    result = yy_vals + a_vals
    
    # Store result to output
    tl.store(output_ptr + offsets, result, mask=mask)

def s421_triton(a, xx, yy):
    """
    Triton implementation of TSVC s421.
    Functionally equivalent to baseline PyTorch version.
    """
    a = a.contiguous()
    xx = xx.contiguous()
    yy = yy.contiguous()
    
    # Clone xx to yy (equivalent to yy = xx)
    yy_new = xx.clone()
    
    n_elements = xx.numel()
    
    # Early return if array too small
    if n_elements <= 1:
        return xx, yy_new
    
    # Create output tensor for xx
    xx_new = torch.empty_like(xx)
    # Copy the last element (unchanged in the operation)
    xx_new[-1] = xx[-1]
    
    # Launch kernel for elements [0, n_elements-2]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements - 1, BLOCK_SIZE),)
    
    s421_kernel[grid](
        a, xx, yy_new, xx_new,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return xx_new, yy_new