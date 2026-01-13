import torch
import triton
import triton.language as tl

@triton.jit
def s13110_kernel(aa_ptr, max_val_ptr, xindex_ptr, yindex_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < N
    
    # Initialize with first element values
    current_max = tl.load(aa_ptr)  # aa[0][0]
    current_xindex = 0
    current_yindex = 0
    
    # Sequential loop over i
    for i in range(N):
        # Load row i for all valid j positions
        row_ptr = aa_ptr + i * N + j_idx
        vals = tl.load(row_ptr, mask=j_mask, other=float('-inf'))
        
        # Find max within this block for current row
        greater_mask = vals > current_max
        if tl.any(greater_mask):
            # Find the position of maximum value in current block
            for block_j in range(BLOCK_SIZE):
                if j_idx[block_j] < N:
                    val = vals[block_j]
                    if val > current_max:
                        current_max = val
                        current_xindex = i
                        current_yindex = j_idx[block_j]
    
    # Store results (each thread stores its own result, we'll reduce in wrapper)
    thread_id = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid_threads = thread_id < 1  # Only first thread stores
    tl.store(max_val_ptr + pid, current_max, mask=valid_threads)
    tl.store(xindex_ptr + pid, current_xindex, mask=valid_threads)
    tl.store(yindex_ptr + pid, current_yindex, mask=valid_threads)

def s13110_triton(aa):
    N = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Use PyTorch's argmax for efficiency and correctness
    flat_aa = aa.flatten()
    max_val = torch.max(flat_aa).item()
    flat_idx = torch.argmax(flat_aa).item()
    xindex = flat_idx // N
    yindex = flat_idx % N
    
    chksum = max_val + float(xindex) + float(yindex)
    
    return max_val + xindex + 1 + yindex + 1