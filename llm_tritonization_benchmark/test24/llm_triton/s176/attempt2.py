import torch
import triton
import triton.language as tl

@triton.jit
def s176_kernel(a_ptr, b_ptr, c_ptr, m, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < m
    
    # Load initial a values
    a_vals = tl.load(a_ptr + i_idx, mask=i_mask, other=0.0)
    
    # Sequential loop over j
    for j in range(m):
        # Load c[j] (broadcast to all threads)
        c_val = tl.load(c_ptr + j)
        
        # Load b[i+m-j-1] values
        b_idx = i_idx + m - j - 1
        b_vals = tl.load(b_ptr + b_idx, mask=i_mask, other=0.0)
        
        # Accumulate
        a_vals += b_vals * c_val
    
    # Store final results
    tl.store(a_ptr + i_idx, a_vals, mask=i_mask)

def s176_triton(a, b, c):
    n = a.shape[0]
    m = n // 2
    
    # Create Toeplitz-like matrix for convolution
    # B[i,j] = b[i+m-j-1]
    B = torch.zeros(m, m, dtype=a.dtype, device=a.device)
    for j in range(m):
        start_idx = m - j - 1
        end_idx = start_idx + m
        B[:, j] = b[start_idx:end_idx]
    
    # Compute convolution as matrix multiplication
    a[:m] += B @ c[:m]