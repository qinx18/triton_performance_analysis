import triton
import triton.language as tl
import torch

@triton.jit
def s3110_kernel(aa_ptr, max_ptr, xindex_ptr, yindex_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # This is a reduction operation to find max element and its indices
    # Each block processes a portion of the 2D array
    
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Initialize local max and indices
    local_max = tl.load(aa_ptr)  # aa[0][0]
    local_xindex = 0
    local_yindex = 0
    
    # Process elements in this block
    for offset in range(BLOCK_SIZE):
        idx = block_start + offset
        if idx < LEN_2D * LEN_2D:
            i = idx // LEN_2D
            j = idx % LEN_2D
            
            # Load current element
            current_val = tl.load(aa_ptr + i * LEN_2D + j)
            
            # Update max and indices if current value is larger
            if current_val > local_max:
                local_max = current_val
                local_xindex = i
                local_yindex = j
    
    # Store results (will need reduction across blocks)
    tl.store(max_ptr + pid, local_max)
    tl.store(xindex_ptr + pid, local_xindex)
    tl.store(yindex_ptr + pid, local_yindex)

def s3110_triton(aa):
    LEN_2D = aa.shape[0]
    total_elements = LEN_2D * LEN_2D
    
    # Choose block size
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(total_elements, BLOCK_SIZE)
    
    # Create temporary buffers for partial results
    max_vals = torch.zeros(num_blocks, dtype=aa.dtype, device=aa.device)
    xindices = torch.zeros(num_blocks, dtype=torch.int32, device=aa.device)
    yindices = torch.zeros(num_blocks, dtype=torch.int32, device=aa.device)
    
    # Launch kernel
    s3110_kernel[(num_blocks,)](
        aa, max_vals, xindices, yindices,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Perform final reduction on CPU to find global max
    max_idx = torch.argmax(max_vals)
    final_max = max_vals[max_idx]
    final_xindex = xindices[max_idx]
    final_yindex = yindices[max_idx]
    
    # Calculate checksum
    chksum = final_max + final_xindex.float() + final_yindex.float()
    
    return final_max + final_xindex + 1 + final_yindex + 1