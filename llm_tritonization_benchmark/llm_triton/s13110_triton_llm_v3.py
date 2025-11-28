import torch
import triton
import triton.language as tl

@triton.jit
def s13110_kernel(aa_ptr, max_ptr, xindex_ptr, yindex_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # This is a reduction kernel that finds the maximum value and its indices
    # We'll use a single thread block to handle the entire reduction
    tid = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Initialize local maximum and indices
    local_max = -float('inf')
    local_xindex = 0
    local_yindex = 0
    
    # Sequential reduction over the 2D array
    for i in range(LEN_2D):
        for j in range(LEN_2D):
            idx = i * LEN_2D + j
            if tid[0] == 0:  # Only first thread does the work
                value = tl.load(aa_ptr + idx)
                if value > local_max:
                    local_max = value
                    local_xindex = i
                    local_yindex = j
    
    # Store results (only first thread)
    if tid[0] == 0:
        tl.store(max_ptr, local_max)
        tl.store(xindex_ptr, local_xindex)
        tl.store(yindex_ptr, local_yindex)

@triton.jit
def s13110_kernel_optimized(aa_ptr, results_ptr, LEN_2D: tl.constexpr):
    # Each program handles a portion of the array
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    
    # Calculate elements per program
    total_elements = LEN_2D * LEN_2D
    elements_per_program = (total_elements + num_programs - 1) // num_programs
    
    start_idx = pid * elements_per_program
    end_idx = tl.minimum(start_idx + elements_per_program, total_elements)
    
    local_max = -float('inf')
    local_xindex = 0
    local_yindex = 0
    
    # Process assigned elements
    for flat_idx in range(start_idx, end_idx):
        i = flat_idx // LEN_2D
        j = flat_idx % LEN_2D
        
        value = tl.load(aa_ptr + flat_idx)
        if value > local_max:
            local_max = value
            local_xindex = i
            local_yindex = j
    
    # Store partial results
    result_offset = pid * 3
    tl.store(results_ptr + result_offset, local_max)
    tl.store(results_ptr + result_offset + 1, local_xindex.to(tl.float32))
    tl.store(results_ptr + result_offset + 2, local_yindex.to(tl.float32))

def s13110_triton(aa):
    LEN_2D = aa.shape[0]
    device = aa.device
    
    # Simple single-threaded approach for correctness
    max_result = torch.zeros(1, dtype=torch.float32, device=device)
    xindex_result = torch.zeros(1, dtype=torch.int32, device=device)
    yindex_result = torch.zeros(1, dtype=torch.int32, device=device)
    
    # Use simple kernel for exact semantics
    grid = (1,)
    s13110_kernel[grid](
        aa, max_result, xindex_result, yindex_result,
        LEN_2D=LEN_2D, BLOCK_SIZE=1
    )
    
    max_val = max_result.item()
    xindex = xindex_result.item()
    yindex = yindex_result.item()
    
    chksum = max_val + float(xindex) + float(yindex)
    
    return max_val + xindex + 1 + yindex + 1