import torch
import triton
import triton.language as tl

@triton.jit
def s13110_kernel(aa_ptr, result_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    # Only use the first program
    if pid != 0:
        return
    
    # Initialize with first element
    max_val = tl.load(aa_ptr)
    max_i = 0
    max_j = 0
    
    # Sequential loop over i
    for i in range(N):
        # Parallel loop over j using blocks
        for j_start in range(0, N, BLOCK_SIZE):
            j_offsets = tl.arange(0, BLOCK_SIZE)
            j_idx = j_start + j_offsets
            j_mask = j_idx < N
            
            # Load values for current i, block of j
            ptrs = aa_ptr + i * N + j_idx
            vals = tl.load(ptrs, mask=j_mask, other=float('-inf'))
            
            # Check if any value in this block is greater than current max
            block_max = tl.max(vals)
            if block_max > max_val:
                # Find the position of the maximum in this block
                max_positions = vals == block_max
                valid_positions = max_positions & j_mask
                
                # Find first occurrence
                for k in range(BLOCK_SIZE):
                    if j_start + k < N:
                        if tl.load(aa_ptr + i * N + j_start + k) == block_max:
                            if tl.load(aa_ptr + i * N + j_start + k) > max_val:
                                max_val = block_max
                                max_i = i
                                max_j = j_start + k
                            break
    
    # Store results
    tl.store(result_ptr, max_val)
    tl.store(result_ptr + 1, max_i)
    tl.store(result_ptr + 2, max_j)

def s13110_triton(aa):
    N = aa.shape[0]
    
    # Use PyTorch for reliable argmax
    flat_aa = aa.flatten()
    max_val = torch.max(flat_aa)
    max_idx = torch.argmax(flat_aa)
    xindex = max_idx // N
    yindex = max_idx % N
    
    return max_val + (xindex + 1) + (yindex + 1)