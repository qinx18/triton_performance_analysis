import torch
import triton
import triton.language as tl

@triton.jit
def s353_kernel(a_ptr, b_ptr, alpha, ip_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID and calculate base offset
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector once - must be power of 2
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process elements in groups of 5
    for i in range(0, BLOCK_SIZE, 5):
        # Calculate indices for this group of 5
        idx0 = block_start + i
        idx1 = idx0 + 1
        idx2 = idx0 + 2
        idx3 = idx0 + 3
        idx4 = idx0 + 4
        
        # Masks for bounds checking
        mask0 = idx0 < n_elements
        mask1 = idx1 < n_elements
        mask2 = idx2 < n_elements
        mask3 = idx3 < n_elements
        mask4 = idx4 < n_elements
        
        # Load indices from ip array
        ip0 = tl.load(ip_ptr + idx0, mask=mask0, other=0)
        ip1 = tl.load(ip_ptr + idx1, mask=mask1, other=0)
        ip2 = tl.load(ip_ptr + idx2, mask=mask2, other=0)
        ip3 = tl.load(ip_ptr + idx3, mask=mask3, other=0)
        ip4 = tl.load(ip_ptr + idx4, mask=mask4, other=0)
        
        # Load values from b array using gathered indices
        b0 = tl.load(b_ptr + ip0, mask=mask0, other=0.0)
        b1 = tl.load(b_ptr + ip1, mask=mask1, other=0.0)
        b2 = tl.load(b_ptr + ip2, mask=mask2, other=0.0)
        b3 = tl.load(b_ptr + ip3, mask=mask3, other=0.0)
        b4 = tl.load(b_ptr + ip4, mask=mask4, other=0.0)
        
        # Load current values from a array
        a0 = tl.load(a_ptr + idx0, mask=mask0, other=0.0)
        a1 = tl.load(a_ptr + idx1, mask=mask1, other=0.0)
        a2 = tl.load(a_ptr + idx2, mask=mask2, other=0.0)
        a3 = tl.load(a_ptr + idx3, mask=mask3, other=0.0)
        a4 = tl.load(a_ptr + idx4, mask=mask4, other=0.0)
        
        # Compute saxpy operations
        result0 = a0 + alpha * b0
        result1 = a1 + alpha * b1
        result2 = a2 + alpha * b2
        result3 = a3 + alpha * b3
        result4 = a4 + alpha * b4
        
        # Store results back to a array
        tl.store(a_ptr + idx0, result0, mask=mask0)
        tl.store(a_ptr + idx1, result1, mask=mask1)
        tl.store(a_ptr + idx2, result2, mask=mask2)
        tl.store(a_ptr + idx3, result3, mask=mask3)
        tl.store(a_ptr + idx4, result4, mask=mask4)

def s353_triton(a, b, c, ip):
    n_elements = a.shape[0]
    alpha = c[0]
    
    # Use power-of-2 block size
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s353_kernel[grid](
        a, b, alpha, ip, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )