import triton
import triton.language as tl
import torch

@triton.jit
def k3mm_kernel(
    A_ptr, B_ptr, C_ptr, D_ptr, E_ptr, F_ptr, G_ptr,
    NI: tl.constexpr, NJ: tl.constexpr, NK: tl.constexpr, 
    NL: tl.constexpr, NM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # E := A*B
    for i in range(NI):
        for j in range(NJ):
            acc = 0.0
            for k in range(NK):
                a_val = tl.load(A_ptr + i * NK + k)
                b_val = tl.load(B_ptr + k * NJ + j)
                acc += a_val * b_val
            tl.store(E_ptr + i * NJ + j, acc)
    
    # F := C*D
    for i in range(NJ):
        for j in range(NL):
            acc = 0.0
            for k in range(NM):
                c_val = tl.load(C_ptr + i * NM + k)
                d_val = tl.load(D_ptr + k * NL + j)
                acc += c_val * d_val
            tl.store(F_ptr + i * NL + j, acc)
    
    # G := E*F
    for i in range(NI):
        for j in range(NL):
            acc = 0.0
            for k in range(NJ):
                e_val = tl.load(E_ptr + i * NJ + k)
                f_val = tl.load(F_ptr + k * NL + j)
                acc += e_val * f_val
            tl.store(G_ptr + i * NL + j, acc)

def k3mm_triton(A, B, C, D, E, F, G, NI, NJ, NK, NL, NM):
    BLOCK_SIZE = 128
    
    grid = (1,)
    
    k3mm_kernel[grid](
        A, B, C, D, E, F, G,
        NI=NI, NJ=NJ, NK=NK, NL=NL, NM=NM,
        BLOCK_SIZE=BLOCK_SIZE
    )