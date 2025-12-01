import triton
import triton.language as tl
import torch

@triton.jit
def s353_kernel(
    a_ptr, b_ptr, ip_ptr, alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and calculate base offset
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset vectors for the 5 unrolled elements
    base_offsets = tl.arange(0, BLOCK_SIZE)
    offsets_0 = block_start + base_offsets
    offsets_1 = offsets_0 + 1
    offsets_2 = offsets_0 + 2
    offsets_3 = offsets_0 + 3
    offsets_4 = offsets_0 + 4
    
    # Create masks for valid elements
    mask_0 = offsets_0 < n_elements
    mask_1 = offsets_1 < n_elements
    mask_2 = offsets_2 < n_elements
    mask_3 = offsets_3 < n_elements
    mask_4 = offsets_4 < n_elements
    
    # Load indices from ip array
    ip_0 = tl.load(ip_ptr + offsets_0, mask=mask_0, other=0)
    ip_1 = tl.load(ip_ptr + offsets_1, mask=mask_1, other=0)
    ip_2 = tl.load(ip_ptr + offsets_2, mask=mask_2, other=0)
    ip_3 = tl.load(ip_ptr + offsets_3, mask=mask_3, other=0)
    ip_4 = tl.load(ip_ptr + offsets_4, mask=mask_4, other=0)
    
    # Load values from b array using gathered indices
    b_0 = tl.load(b_ptr + ip_0, mask=mask_0, other=0.0)
    b_1 = tl.load(b_ptr + ip_1, mask=mask_1, other=0.0)
    b_2 = tl.load(b_ptr + ip_2, mask=mask_2, other=0.0)
    b_3 = tl.load(b_ptr + ip_3, mask=mask_3, other=0.0)
    b_4 = tl.load(b_ptr + ip_4, mask=mask_4, other=0.0)
    
    # Load current values from a array
    a_0 = tl.load(a_ptr + offsets_0, mask=mask_0, other=0.0)
    a_1 = tl.load(a_ptr + offsets_1, mask=mask_1, other=0.0)
    a_2 = tl.load(a_ptr + offsets_2, mask=mask_2, other=0.0)
    a_3 = tl.load(a_ptr + offsets_3, mask=mask_3, other=0.0)
    a_4 = tl.load(a_ptr + offsets_4, mask=mask_4, other=0.0)
    
    # Perform saxpy operations
    a_0 += alpha * b_0
    a_1 += alpha * b_1
    a_2 += alpha * b_2
    a_3 += alpha * b_3
    a_4 += alpha * b_4
    
    # Store results back to a array
    tl.store(a_ptr + offsets_0, a_0, mask=mask_0)
    tl.store(a_ptr + offsets_1, a_1, mask=mask_1)
    tl.store(a_ptr + offsets_2, a_2, mask=mask_2)
    tl.store(a_ptr + offsets_3, a_3, mask=mask_3)
    tl.store(a_ptr + offsets_4, a_4, mask=mask_4)

def s353_triton(a, b, c, ip):
    n_elements = a.shape[0]
    alpha = c[0].item()
    
    # Calculate grid size - each block processes BLOCK_SIZE elements with unroll factor of 5
    BLOCK_SIZE = 256
    # Need to handle 5 elements per iteration, so adjust accordingly
    grid = ((n_elements + 4) // 5 + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    s353_kernel[(grid,)](
        a, b, ip, alpha,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )