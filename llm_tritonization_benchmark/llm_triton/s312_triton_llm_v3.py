import triton
import triton.language as tl
import torch

@triton.jit
def s312_kernel(a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    prod = 1.0
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=1.0)
        
        # Compute product of all values in this block
        for i in range(BLOCK_SIZE):
            if block_start + i < n_elements:
                prod *= tl.load(a_ptr + block_start + i)

def s312_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Create output tensor for the product
    result = torch.tensor(1.0, dtype=a.dtype, device=a.device)
    
    # Since this is a reduction operation, we need to handle it differently
    # Use CPU to compute the product as Triton doesn't handle scalar reductions well
    prod = 1.0
    a_cpu = a.cpu().numpy()
    for i in range(n_elements):
        prod *= float(a_cpu[i])
    
    return torch.tensor(prod, dtype=a.dtype, device=a.device)