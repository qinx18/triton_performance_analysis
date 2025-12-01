import torch
import triton
import triton.language as tl

@triton.jit
def s292_kernel(
    a_ptr, b_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s292 - sliding window computation with wraparound indices.
    Each thread processes one element, computing a[i] = (b[i] + b[im1] + b[im2]) * 0.333
    where im1 and im2 are the previous two indices in the sliding window pattern.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # For each position i, we need to compute im1 and im2 based on the sliding pattern
    # im1 starts at n-1, then becomes 0, 1, 2, ..., i-1
    # im2 starts at n-2, then becomes n-1, 0, 1, ..., i-2
    
    # Load current b values
    b_i = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Compute im1 indices: for i=0: im1=n-1, for i=1: im1=0, for i>=1: im1=i-1
    im1_indices = tl.where(offsets == 0, n_elements - 1, offsets - 1)
    
    # Compute im2 indices: for i=0: im2=n-2, for i=1: im2=n-1, for i>=2: im2=i-2
    im2_indices = tl.where(offsets == 0, n_elements - 2,
                          tl.where(offsets == 1, n_elements - 1, offsets - 2))
    
    # Load b[im1] and b[im2] values
    b_im1 = tl.load(b_ptr + im1_indices, mask=mask, other=0.0)
    b_im2 = tl.load(b_ptr + im2_indices, mask=mask, other=0.0)
    
    # Compute result: (b[i] + b[im1] + b[im2]) * 0.333
    result = (b_i + b_im1 + b_im2) * 0.333
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s292_triton(a, b, im1, im2):
    """
    Triton implementation of s292 - sliding window computation with wraparound indices.
    Parallelizes the computation across elements using appropriate block sizing.
    """
    n_elements = b.numel()
    
    # Choose block size based on problem size for optimal memory coalescing
    BLOCK_SIZE = triton.next_power_of_2(min(1024, n_elements))
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Ensure tensors are contiguous for optimal memory access
    a = a.contiguous()
    b = b.contiguous()
    
    # Launch kernel
    s292_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a