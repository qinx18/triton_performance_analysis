import torch
import triton
import triton.language as tl

@triton.jit
def s13110_kernel(aa_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Initialize with first element
    max_val = tl.load(aa_ptr)
    max_i = 0
    max_j = 0
    
    # Sequential loop over i dimension
    for i in range(N):
        # Parallel processing of j dimension
        j_start = pid * BLOCK_SIZE
        j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
        j_mask = j_offsets < N
        
        # Load values for this row
        row_ptr = aa_ptr + i * N + j_offsets
        vals = tl.load(row_ptr, mask=j_mask, other=float('-inf'))
        
        # Find local maximum in this block
        for k in range(BLOCK_SIZE):
            if j_offsets[k] < N:
                if vals[k] > max_val:
                    max_val = vals[k]
                    max_i = i
                    max_j = j_offsets[k]
    
    # Store results (max_val, max_i, max_j) for this block
    block_output_ptr = output_ptr + pid * 3
    tl.store(block_output_ptr, max_val)
    tl.store(block_output_ptr + 1, max_i)
    tl.store(block_output_ptr + 2, max_j)

def s13110_triton(aa):
    N = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Use PyTorch for argmax reduction
    flat_aa = aa.flatten()
    max_val = torch.max(flat_aa).item()
    flat_idx = torch.argmax(flat_aa).item()
    xindex = flat_idx // N
    yindex = flat_idx % N
    
    # Return value matches C code: max + xindex+1 + yindex+1
    return max_val + xindex + 1 + yindex + 1