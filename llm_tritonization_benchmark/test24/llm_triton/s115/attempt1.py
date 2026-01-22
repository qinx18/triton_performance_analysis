import torch
import triton
import triton.language as tl

@triton.jit
def s115_kernel(a_ptr, aa_ptr, j, N, BLOCK_SIZE: tl.constexpr):
    # Calculate starting index for this j
    start_i = j + 1
    
    # Block processing
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE + start_i
    
    # Generate offsets
    offsets = tl.arange(0, BLOCK_SIZE)
    i_indices = block_start + offsets
    
    # Mask for valid i values
    mask = i_indices < N
    
    # Load a[j] (scalar broadcast)
    a_j = tl.load(a_ptr + j)
    
    # Load aa[j][i] values
    aa_offsets = j * N + i_indices
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=mask, other=0.0)
    
    # Load a[i] values
    a_vals = tl.load(a_ptr + i_indices, mask=mask, other=0.0)
    
    # Compute: a[i] -= aa[j][i] * a[j]
    new_vals = a_vals - aa_vals * a_j
    
    # Store back to a[i]
    tl.store(a_ptr + i_indices, new_vals, mask=mask)

def s115_triton(a, aa):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Sequential loop over j
    for j in range(N):
        # Number of elements to process for this j
        num_elements = N - (j + 1)
        if num_elements <= 0:
            continue
            
        # Grid size
        grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
        
        # Launch kernel
        s115_kernel[grid](a, aa, j, N, BLOCK_SIZE)