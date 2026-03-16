import triton
import triton.language as tl
import torch

@triton.jit
def fdtd_2d_step1_kernel(fict_ptr, ey_ptr, t: tl.constexpr, NY: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_start = pid * BLOCK_SIZE
    j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < NY
    
    fict_val = tl.load(fict_ptr + t)
    ey_indices = j_offsets
    tl.store(ey_ptr + ey_indices, fict_val, mask=j_mask)

@triton.jit
def fdtd_2d_step2_kernel(ey_ptr, hz_ptr, NX: tl.constexpr, NY: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    total_elements = (NX - 1) * NY
    elem_start = pid * BLOCK_SIZE
    elem_offsets = elem_start + tl.arange(0, BLOCK_SIZE)
    elem_mask = elem_offsets < total_elements
    
    i = (elem_offsets // NY) + 1
    j = elem_offsets % NY
    
    ey_idx = i * NY + j
    hz_idx = i * NY + j
    hz_prev_idx = (i - 1) * NY + j
    
    ey_val = tl.load(ey_ptr + ey_idx, mask=elem_mask)
    hz_val = tl.load(hz_ptr + hz_idx, mask=elem_mask)
    hz_prev_val = tl.load(hz_ptr + hz_prev_idx, mask=elem_mask)
    
    new_ey = ey_val - 0.5 * (hz_val - hz_prev_val)
    tl.store(ey_ptr + ey_idx, new_ey, mask=elem_mask)

@triton.jit
def fdtd_2d_step3_kernel(ex_ptr, hz_ptr, NX: tl.constexpr, NY: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    total_elements = NX * (NY - 1)
    elem_start = pid * BLOCK_SIZE
    elem_offsets = elem_start + tl.arange(0, BLOCK_SIZE)
    elem_mask = elem_offsets < total_elements
    
    i = elem_offsets // (NY - 1)
    j = (elem_offsets % (NY - 1)) + 1
    
    ex_idx = i * NY + j
    hz_idx = i * NY + j
    hz_prev_idx = i * NY + (j - 1)
    
    ex_val = tl.load(ex_ptr + ex_idx, mask=elem_mask)
    hz_val = tl.load(hz_ptr + hz_idx, mask=elem_mask)
    hz_prev_val = tl.load(hz_ptr + hz_prev_idx, mask=elem_mask)
    
    new_ex = ex_val - 0.5 * (hz_val - hz_prev_val)
    tl.store(ex_ptr + ex_idx, new_ex, mask=elem_mask)

@triton.jit
def fdtd_2d_step4_kernel(ex_ptr, ey_ptr, hz_ptr, NX: tl.constexpr, NY: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    total_elements = (NX - 1) * (NY - 1)
    elem_start = pid * BLOCK_SIZE
    elem_offsets = elem_start + tl.arange(0, BLOCK_SIZE)
    elem_mask = elem_offsets < total_elements
    
    i = elem_offsets // (NY - 1)
    j = elem_offsets % (NY - 1)
    
    hz_idx = i * NY + j
    ex_idx = i * NY + j
    ex_next_idx = i * NY + (j + 1)
    ey_idx = i * NY + j
    ey_next_idx = (i + 1) * NY + j
    
    hz_val = tl.load(hz_ptr + hz_idx, mask=elem_mask)
    ex_val = tl.load(ex_ptr + ex_idx, mask=elem_mask)
    ex_next_val = tl.load(ex_ptr + ex_next_idx, mask=elem_mask)
    ey_val = tl.load(ey_ptr + ey_idx, mask=elem_mask)
    ey_next_val = tl.load(ey_ptr + ey_next_idx, mask=elem_mask)
    
    new_hz = hz_val - 0.7 * (ex_next_val - ex_val + ey_next_val - ey_val)
    tl.store(hz_ptr + hz_idx, new_hz, mask=elem_mask)

def fdtd_2d_triton(_fict_, ex, ey, hz, NX, NY, TMAX):
    BLOCK_SIZE = 256
    
    for t in range(TMAX):
        # Step 1
        grid1 = (triton.cdiv(NY, BLOCK_SIZE),)
        fdtd_2d_step1_kernel[grid1](_fict_, ey, t, NY, BLOCK_SIZE)
        
        # Step 2
        total_elements2 = (NX - 1) * NY
        grid2 = (triton.cdiv(total_elements2, BLOCK_SIZE),)
        fdtd_2d_step2_kernel[grid2](ey, hz, NX, NY, BLOCK_SIZE)
        
        # Step 3
        total_elements3 = NX * (NY - 1)
        grid3 = (triton.cdiv(total_elements3, BLOCK_SIZE),)
        fdtd_2d_step3_kernel[grid3](ex, hz, NX, NY, BLOCK_SIZE)
        
        # Step 4
        total_elements4 = (NX - 1) * (NY - 1)
        grid4 = (triton.cdiv(total_elements4, BLOCK_SIZE),)
        fdtd_2d_step4_kernel[grid4](ex, ey, hz, NX, NY, BLOCK_SIZE)