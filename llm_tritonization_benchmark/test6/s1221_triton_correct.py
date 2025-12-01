import torch
import triton
import triton.language as tl

@triton.jit
def s1221_kernel_chain(
    a_ptr, b_ptr, n,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Process one chain with stride-4 pattern using in-kernel sequential loop.

    Program 0: processes chain 0 → b[4], b[8], b[12], ...
    Program 1: processes chain 1 → b[5], b[9], b[13], ...
    Program 2: processes chain 2 → b[6], b[10], b[14], ...
    Program 3: processes chain 3 → b[7], b[11], b[15], ...

    Each chain is sequential: b[i] = b[i-4] + a[i]
    """
    # Each program processes one chain
    chain_id = tl.program_id(axis=0)

    # Load base value (never modified)
    base_idx = chain_id
    accumulator = tl.load(b_ptr + base_idx)

    # Process chain sequentially
    idx = chain_id + 4
    while idx < n:
        # Load a[idx]
        a_val = tl.load(a_ptr + idx)

        # Update accumulator
        accumulator = accumulator + a_val

        # Store to b[idx]
        tl.store(b_ptr + idx, accumulator)

        # Move to next element in chain (stride-4)
        idx += 4

def s1221_triton(a, b):
    """
    Corrected Triton implementation of TSVC s1221

    Strategy: 4 independent chains (stride-4 pattern).
    Launch 4 programs in parallel, each processing one chain sequentially.

    Chain 0: b[4], b[8], b[12], ... (program 0)
    Chain 1: b[5], b[9], b[13], ... (program 1)
    Chain 2: b[6], b[10], b[14], ... (program 2)
    Chain 3: b[7], b[11], b[15], ... (program 3)
    """
    a = a.contiguous()
    b = b.contiguous()

    n = b.shape[0]

    if n < 5:
        return b

    # Launch 4 programs in parallel, each processes one chain
    grid = (4,)
    BLOCK_SIZE = 1

    s1221_kernel_chain[grid](
        a, b, n,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return b
