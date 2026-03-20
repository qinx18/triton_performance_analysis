import triton
import triton.language as tl
import torch

@triton.jit
def cholesky_kernel_phase1(A_ptr, i: tl.constexpr, j: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    k_offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = (k_offsets < j) & (block_start < j)
    
    if mask.any():
        i_k_offsets = i * N + k_offsets
        j_k_offsets = j * N + k_offsets
        
        a_i_k = tl.load(A_ptr + i_k_offsets, mask=mask, other=0.0)
        a_j_k = tl.load(A_ptr + j_k_offsets, mask=mask, other=0.0)
        
        products = a_i_k * a_j_k
        reduction = tl.sum(products)
        
        if pid == 0:
            i_j_offset = i * N + j
            a_i_j = tl.load(A_ptr + i_j_offset)
            a_i_j = a_i_j - reduction
            tl.store(A_ptr + i_j_offset, a_i_j)

@triton.jit
def cholesky_kernel_phase2(A_ptr, i: tl.constexpr, j: tl.constexpr, N: tl.constexpr):
    i_j_offset = i * N + j
    j_j_offset = j * N + j
    
    a_i_j = tl.load(A_ptr + i_j_offset)
    a_j_j = tl.load(A_ptr + j_j_offset)
    
    a_i_j = a_i_j / a_j_j
    tl.store(A_ptr + i_j_offset, a_i_j)

@triton.jit
def cholesky_kernel_phase3(A_ptr, i: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    k_offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = (k_offsets < i) & (block_start < i)
    
    if mask.any():
        i_k_offsets = i * N + k_offsets
        a_i_k = tl.load(A_ptr + i_k_offsets, mask=mask, other=0.0)
        
        squares = a_i_k * a_i_k
        reduction = tl.sum(squares)
        
        if pid == 0:
            i_i_offset = i * N + i
            a_i_i = tl.load(A_ptr + i_i_offset)
            a_i_i = a_i_i - reduction
            tl.store(A_ptr + i_i_offset, a_i_i)

@triton.jit
def cholesky_kernel_phase4(A_ptr, i: tl.constexpr, N: tl.constexpr):
    i_i_offset = i * N + i
    a_i_i = tl.load(A_ptr + i_i_offset)
    a_i_i = tl.sqrt(a_i_i)
    tl.store(A_ptr + i_i_offset, a_i_i)

def cholesky_triton(A, N):
    BLOCK_SIZE = 64
    
    for i in range(N):
        for j in range(i):
            if j > 0:
                grid = (triton.cdiv(j, BLOCK_SIZE),)
                cholesky_kernel_phase1[grid](A, i, j, N, BLOCK_SIZE)
            
            grid = (1,)
            cholesky_kernel_phase2[grid](A, i, j, N)
        
        if i > 0:
            grid = (triton.cdiv(i, BLOCK_SIZE),)
            cholesky_kernel_phase3[grid](A, i, N, BLOCK_SIZE)
        
        grid = (1,)
        cholesky_kernel_phase4[grid](A, i, N)