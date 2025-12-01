"""
Generate Triton implementations using Claude API from original TSVC C code.

Uses the original TSVC C functions as baseline, with PET analysis for:
1. Flow dependency (RAW) analysis
2. WAR anti-dependency analysis
3. Parallelization dimension analysis (which dims can be parallel vs sequential)
"""

import anthropic
import os
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

# Add PET analysis directory to path
sys.path.insert(0, "/home/qinxiao/workspace/pet/isl_analysis")
try:
    from compute_war_dependences import analyze_kernel_war
    HAS_WAR_ANALYSIS = True
except ImportError:
    HAS_WAR_ANALYSIS = False
    analyze_kernel_war = None
    print("Warning: WAR analysis not available (islpy not installed)")

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=API_KEY)

# Paths
TSVC_SOURCE = "/home/qinxiao/workspace/TSVC_2/src/archive/tsvc_orig.c"
KERNELS_DIR = "/home/qinxiao/workspace/pet/isl_analysis/kernels"
PARALLEL_DIMS_ANALYSIS_FILE = "/home/qinxiao/workspace/pet/isl_analysis/results/parallel_dims_analysis.txt"
BASELINES_DIR = Path("baselines")  # Still needed for function signature reference


def extract_tsvc_function(func_name: str) -> Optional[dict]:
    """Extract a function from the original TSVC source file."""
    with open(TSVC_SOURCE, 'r') as f:
        content = f.read()

    # Find function definition
    pattern = rf'real_t {func_name}\s*\(struct args_t \* func_args\)\s*\{{'
    match = re.search(pattern, content)
    if not match:
        return None

    start = match.start()

    # Find matching closing brace
    brace_count = 0
    in_func = False
    end = start

    for i, char in enumerate(content[start:], start):
        if char == '{':
            brace_count += 1
            in_func = True
        elif char == '}':
            brace_count -= 1
            if in_func and brace_count == 0:
                end = i + 1
                break

    func_body = content[start:end]

    # Extract local variables
    local_vars = extract_local_variables(func_body)

    # Extract the kernel loop (inner part without nl loop)
    kernel_loop = extract_kernel_loop(func_body)

    # Extract comment/description
    desc_match = re.search(rf'//.*?\n.*?real_t {func_name}', content[max(0,start-200):start])
    description = ""
    if desc_match:
        desc_lines = desc_match.group(0).split('\n')
        description = '\n'.join(line for line in desc_lines if line.strip().startswith('//'))

    return {
        'name': func_name,
        'full_body': func_body,
        'local_vars': local_vars,
        'kernel_loop': kernel_loop,
        'description': description
    }


def extract_local_variables(func_body: str) -> List[str]:
    """Extract local variable declarations from function body."""
    variables = []
    var_patterns = [
        r'^\s*(int\s+(?!nl\b)(?!i\b)(?!j\b)(?!k\b)\w+(?:\s*,\s*\w+)*\s*(?:=\s*[^;]+)?)\s*;',
        r'^\s*(real_t\s+\w+(?:\s*,\s*\w+)*\s*(?:=\s*[^;]+)?)\s*;',
        r'^\s*(float\s+\w+(?:\s*,\s*\w+)*\s*(?:=\s*[^;]+)?)\s*;',
        r'^\s*(double\s+\w+(?:\s*,\s*\w+)*\s*(?:=\s*[^;]+)?)\s*;',
    ]

    lines = func_body.split('\n')
    for line in lines:
        if 'for' in line and ('nl' in line or 'i =' in line or 'j =' in line):
            continue
        if 'gettimeofday' in line:
            continue

        for pattern in var_patterns:
            match = re.match(pattern, line)
            if match:
                decl = match.group(1).strip()
                variables.append(decl)
                break

    return variables


def extract_kernel_loop(func_body: str) -> str:
    """Extract the kernel loop from function body (excluding outer nl loop)."""
    lines = func_body.split('\n')
    loop_lines = []
    in_main_loop = False
    brace_depth = 0

    for line in lines:
        stripped = line.strip()

        # Skip initialization and timing code
        if 'initialise_arrays' in stripped or 'gettimeofday' in stripped:
            continue
        if 'dummy(' in stripped or 'calc_checksum' in stripped:
            continue
        if stripped.startswith('//'):
            continue

        # Check for the nl iteration loop
        if 'for' in stripped and 'nl' in stripped:
            in_main_loop = True
            if '{' in stripped:
                brace_depth = 1
                remainder = stripped[stripped.find('{')+1:].strip()
                if remainder:
                    loop_lines.append(remainder)
            continue

        if in_main_loop:
            for char in stripped:
                if char == '{':
                    brace_depth += 1
                elif char == '}':
                    brace_depth -= 1

            if brace_depth > 0:
                if 'dummy(' not in stripped:
                    loop_lines.append(line)
            else:
                break

    return '\n'.join(loop_lines)


def get_pytorch_signature(kernel_name: str) -> Optional[str]:
    """Get the function signature from the PyTorch baseline for reference."""
    baseline_file = BASELINES_DIR / f"{kernel_name}_baseline.py"
    if not baseline_file.exists():
        return None

    with open(baseline_file, 'r') as f:
        content = f.read()

    # Find function definition
    match = re.search(rf'def {kernel_name}_pytorch\((.*?)\):', content)
    if match:
        return match.group(1)
    return None


def load_parallelization_analysis(kernel_name: str) -> Optional[dict]:
    """Load parallelization analysis for a kernel from the pre-computed analysis file."""
    if not os.path.exists(PARALLEL_DIMS_ANALYSIS_FILE):
        return None

    with open(PARALLEL_DIMS_ANALYSIS_FILE, 'r') as f:
        content = f.read()

    eq40 = '=' * 40
    eq80 = '=' * 80
    pattern = rf'{re.escape(eq40)}\nKernel: {re.escape(kernel_name)}\n{re.escape(eq40)}\n(.*?)(?=\n{re.escape(eq40)}\nKernel:|\n\n{re.escape(eq80)}|$)'
    match = re.search(pattern, content, re.DOTALL)

    if not match:
        return None

    section = match.group(1).strip()

    result = {
        'kernel': kernel_name,
        'c_code': '',
        'dims': [],
        'is_triangular': False,
        'triangular_info': None,
        'self_dependencies': [],
        'options': [],
        'summary': ''
    }

    # Extract C code
    c_match = re.search(r'C Code: (.*?)(?=\n\nStatement|\n\n|$)', section, re.DOTALL)
    if c_match:
        result['c_code'] = c_match.group(1).strip()

    # Extract dimensions
    dims_match = re.search(r'Dimensions: \[(.*?)\]', section)
    if dims_match:
        result['dims'] = [d.strip().strip("'") for d in dims_match.group(1).split(',')]

    # Check triangular
    tri_match = re.search(r'Triangular bounds: (\w+) < (\w+)', section)
    if tri_match:
        result['is_triangular'] = True
        result['triangular_info'] = {'smaller': tri_match.group(1), 'larger': tri_match.group(2)}

    # Extract self-dependencies
    deps_match = re.findall(r'- (\w+): write \[(.*?)\], read \[(.*?)\]', section)
    for arr, write_expr, read_expr in deps_match:
        result['self_dependencies'].append({
            'array': arr,
            'write_expr': write_expr,
            'read_expr': read_expr
        })

    # Extract parallelization options
    opts_section = re.search(r'Parallelization Options:(.*?)(?=\n\n  SUMMARY|\n\n\n|\n\n$)', section, re.DOTALL)
    if opts_section:
        opt_lines = opts_section.group(1).split('\n')
        current_opt = None

        for line in opt_lines:
            opt_match = re.match(r'\s*(\w+)-sequential,\s*(\w+)-parallel:\s*(VALID|INVALID)\s*\((\w+)\)', line)
            if opt_match:
                current_opt = {
                    'sequential_dim': opt_match.group(1),
                    'parallel_dim': opt_match.group(2),
                    'valid': opt_match.group(3) == 'VALID',
                    'parallelism_type': opt_match.group(4),
                    'triton_strategy': None,
                    'issues': [],
                    'explanations': []
                }
                result['options'].append(current_opt)
            elif current_opt and 'Triton Strategy:' in line:
                strat_match = re.search(r'Triton Strategy:\s*(\S+)', line)
                if strat_match:
                    current_opt['triton_strategy'] = strat_match.group(1)
            elif current_opt and '!!' in line:
                issue_text = line.split('!!', 1)[1].strip() if '!!' in line else line.strip()
                current_opt['issues'].append(issue_text)
            elif current_opt and line.strip() and not line.strip().startswith('Parallelization'):
                current_opt['explanations'].append(line.strip())

    summary_match = re.search(r'SUMMARY: (.+)', section)
    if summary_match:
        result['summary'] = summary_match.group(1).strip()

    return result


def load_war_analysis(kernel_name: str) -> Optional[dict]:
    """Load WAR anti-dependency analysis for a kernel."""
    if not HAS_WAR_ANALYSIS or analyze_kernel_war is None:
        return None

    kernel_file = os.path.join(KERNELS_DIR, f"{kernel_name}.c")
    if not os.path.exists(kernel_file):
        return None

    try:
        return analyze_kernel_war(kernel_file)
    except Exception as e:
        print(f"  Warning: WAR analysis failed: {e}")
        return None


def build_war_instructions(kernel_name: str, war_result: dict) -> str:
    """Build specific instructions for handling WAR dependencies."""
    if not war_result or war_result['parallelization_safe']:
        return ""

    arrays_to_copy = war_result['arrays_needing_copy']

    instructions = f"""
## CRITICAL: WAR Race Condition Handling Required

This kernel has WAR (Write-After-Read) anti-dependencies that cause race conditions in parallel execution.
**Arrays requiring read-only copy**: {arrays_to_copy}

### Problem
In parallel execution across GPU blocks, if block B writes to a location before block A reads from it,
block A reads the wrong (modified) value instead of the original value.

### Required Solution: Read-Only Copy Pattern
Pass a **read-only copy** of the array to the kernel. All threads load from the copy (immutable)
and store results to the original array.

### Implementation Template
```python
# In wrapper function - create read-only copy BEFORE launching kernel:
def {kernel_name}_triton(...):
"""
    for arr in arrays_to_copy:
        instructions += f"    {arr}_copy = {arr}.clone()  # Read-only copy\n"

    instructions += f"""
    # Pass BOTH original (for writes) AND copy (for reads) to kernel
    {kernel_name}_kernel[grid](
"""
    for arr in arrays_to_copy:
        instructions += f"        {arr},        # Write destination (original)\n"
        instructions += f"        {arr}_copy,   # Read source (immutable copy)\n"

    instructions += """        ...other_args...
    )

# In kernel - LOAD from copy, STORE to original:
@triton.jit
def {kernel_name}_kernel(
"""
    for arr in arrays_to_copy:
        instructions += f"    {arr}_ptr,        # Write destination\n"
        instructions += f"    {arr}_copy_ptr,   # Read source\n"

    instructions += """    ...
):
    offsets = ...
"""
    for arr in arrays_to_copy:
        instructions += f"    {arr}_vals = tl.load({arr}_copy_ptr + offsets, mask=mask)  # Read from COPY\n"

    instructions += """
    result = ...  # Compute using loaded values

"""
    for arr in arrays_to_copy:
        instructions += f"    tl.store({arr}_ptr + write_offsets, result, mask=mask)  # Write to ORIGINAL\n"

    instructions += """```

**Why this works**: The copy preserves original values for all threads. Reads from immutable copy,
writes to original - no conflict possible.

**CRITICAL: Use forward iteration**
ALWAYS use FORWARD iteration with ascending offsets:
```python
# CORRECT - Forward iteration
offsets = block_start + tl.arange(0, BLOCK_SIZE)

# INCORRECT - Reverse iteration (causes memory access issues in Triton)
# offsets = block_start - tl.arange(0, BLOCK_SIZE)  # DO NOT USE
```
"""
    return instructions


def build_parallelization_instructions(kernel_name: str, analysis: Optional[dict]) -> str:
    """Build specific instructions for parallelization strategy."""
    if not analysis:
        return ""

    valid_options = [opt for opt in analysis['options'] if opt['valid']]
    invalid_options = [opt for opt in analysis['options'] if not opt['valid']]

    if not valid_options and not invalid_options:
        return ""

    lines = []
    lines.append("")
    lines.append("## CRITICAL: Parallelization Strategy")
    lines.append("")
    lines.append(f"**Loop structure**: `{analysis['c_code']}`")
    lines.append(f"**Dimensions**: {analysis['dims']}")
    if analysis['is_triangular']:
        tri = analysis['triangular_info']
        lines.append(f"**Triangular bounds**: {tri['smaller']} < {tri['larger']}")
    lines.append("")

    if analysis['self_dependencies']:
        lines.append("**Data dependencies detected**:")
        for dep in analysis['self_dependencies']:
            lines.append(f"- Array `{dep['array']}`: write to `[{dep['write_expr']}]`, read from `[{dep['read_expr']}]`")
        lines.append("")

    # Determine the recommended strategy
    if len(valid_options) >= 1:
        opt = valid_options[0]
        seq_dim = opt['sequential_dim']
        par_dim = opt['parallel_dim']
        par_type = opt['parallelism_type']

        lines.append(f"### Required Strategy: {seq_dim}-sequential, {par_dim}-parallel")
        lines.append("")
        lines.append("**Implementation Pattern:**")
        lines.append(f"- Python wrapper: loop over `{seq_dim}` sequentially (one kernel launch per {seq_dim} value)")
        lines.append(f"- Triton kernel: parallelize ALL `{par_dim}` values using VECTORIZED operations")
        lines.append("")
        lines.append("**CRITICAL: Use vectorized operations inside the kernel!**")
        lines.append("- Use `tl.arange(0, BLOCK_SIZE)` to create index vectors")
        lines.append("- Use vectorized `tl.load` and `tl.store` with masks")
        lines.append("- NEVER use Python-style `for` loops inside @triton.jit kernels - they run serially!")
        lines.append("")

        if par_type == 'reduction':
            lines.append(f"**Reduction pattern:** Use `tl.sum()` to reduce across {par_dim} dimension within each block")
            lines.append("")

        lines.append("**Correct example:**")
        lines.append("```python")
        lines.append("def kernel_triton(...):")
        lines.append(f"    for {seq_dim}_val in range(...):  # Sequential in Python - OK")
        lines.append(f"        grid = (triton.cdiv(N, BLOCK_SIZE),)")
        lines.append(f"        kernel[grid](..., {seq_dim}_val)")
        lines.append("")
        lines.append("@triton.jit")
        lines.append(f"def kernel(..., {seq_dim}_val, BLOCK_SIZE: tl.constexpr):")
        lines.append(f"    # Fully vectorized over {par_dim}")
        lines.append(f"    pid = tl.program_id(0)")
        lines.append(f"    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)")
        lines.append(f"    mask = offsets < N")
        lines.append(f"    vals = tl.load(ptr + offsets, mask=mask)  # Vectorized load")
        lines.append(f"    result = vals * 2  # Vectorized compute")
        lines.append(f"    tl.store(out_ptr + offsets, result, mask=mask)  # Vectorized store")
        lines.append("```")
        lines.append("")

    if invalid_options:
        lines.append("### INVALID Parallelization (DO NOT USE)")
        for opt in invalid_options:
            lines.append(f"**{opt['sequential_dim']}-sequential, {opt['parallel_dim']}-parallel**: INCORRECT - causes race conditions")
        lines.append("")

    return "\n".join(lines)


def generate_triton_from_tsvc(kernel_name: str) -> tuple:
    """Generate Triton implementation from original TSVC C code."""

    # Extract original C function
    tsvc_func = extract_tsvc_function(kernel_name)
    if not tsvc_func:
        raise ValueError(f"Could not find function {kernel_name} in TSVC source")

    # Get PyTorch signature for reference
    pytorch_sig = get_pytorch_signature(kernel_name)

    # Load WAR analysis
    war_result = load_war_analysis(kernel_name)
    war_section = build_war_instructions(kernel_name, war_result)

    # Load parallelization analysis
    par_analysis = load_parallelization_analysis(kernel_name)
    par_section = build_parallelization_instructions(kernel_name, par_analysis)

    # Build the kernel loop with local variables
    c_code_section = tsvc_func['kernel_loop']
    if tsvc_func['local_vars']:
        local_vars_str = '\n'.join(f"    {v};" for v in tsvc_func['local_vars'])
        c_code_section = f"// Local variables:\n{local_vars_str}\n\n// Kernel loop:\n{c_code_section}"

    # Build prompt
    prompt = f"""I have an original TSVC (Test Suite for Vectorizing Compilers) C function that I want to implement in Triton for GPU acceleration.

## Original TSVC C Code:
```c
{tsvc_func['description']}
{tsvc_func['full_body']}
```

## Kernel Loop to Implement:
```c
{c_code_section}
```
{war_section}{par_section}

## Array Information:
- Arrays `a`, `b`, `c`, `d`, `e` are 1D float arrays of size LEN_1D (typically 32000)
- Arrays `aa`, `bb`, `cc`, `tt` are 2D float arrays of size LEN_2D x LEN_2D (typically 256x256)
- `flat_2d_array` is a 1D float array of size LEN_2D*LEN_2D
- `indx` is a 1D int array of size LEN_1D

## Requirements:
Please generate a complete Triton implementation that:
1. Includes a @triton.jit kernel function named `{kernel_name}_kernel`
2. Includes a Python wrapper function named `{kernel_name}_triton`
3. The wrapper should accept ONLY the PyTorch tensor arrays used in the computation (e.g., a, b, c)
4. Uses appropriate block sizes and memory access patterns
5. Handles edge cases with masking
6. Is functionally equivalent to the C code (same computation, same results)

## CRITICAL: Function Signature Requirements
**DO NOT include** the `iterations` parameter or the outer `for (int nl = ...)` timing loop in your implementation.
The `iterations` parameter is only used for benchmarking timing in the original C code - it should NOT be part of the Triton function.

**Correct example**:
```python
def {kernel_name}_triton(a, b):  # Only array parameters
    ...  # Just the kernel computation, NO timing loop
```

**Incorrect example**:
```python
def {kernel_name}_triton(a, b, iterations):  # WRONG: No iterations parameter
    for _ in range(iterations):  # WRONG: No timing loop
        ...
```
"""

    prompt += """
IMPORTANT:
- The function should only accept array tensor parameters (a, b, c, etc.) - NO iterations parameter
- Do NOT implement the outer timing loop (for nl = 0; nl < iterations; nl++)
- If WAR dependencies are shown above, you MUST use the read-only copy pattern
- If parallelization analysis is shown above, you MUST follow the specified parallelization strategy
- DO NOT parallelize dimensions marked as INVALID - this will cause race conditions
- Use forward iteration (ascending indices) for memory access patterns

## CRITICAL: Triton Compilation Rule

**NEVER use `tl.arange()` inside a for loop - it causes compilation errors:**
```python
# ❌ WRONG - causes compilation error
for block_start in range(0, n, BLOCK_SIZE):
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # ERROR!

# ✅ CORRECT - define tl.arange() ONCE at kernel start, before any loops
offsets = tl.arange(0, BLOCK_SIZE)  # Define once at start
for block_start in range(0, n, BLOCK_SIZE):
    current_offsets = block_start + offsets  # Reuse the pre-defined offsets
```

**When using tensors as indices, they must be integer type (long, int) - convert float tensors with `.to(tl.int32)` or `.to(torch.long)`**

Provide ONLY the Python code, no additional explanation."""

    print(f"Generating Triton code for {kernel_name}...")
    if war_result and not war_result['parallelization_safe']:
        print(f"  ⚠ WAR dependencies detected for arrays: {war_result['arrays_needing_copy']}")
    if par_analysis:
        valid_opts = [opt for opt in par_analysis['options'] if opt['valid']]
        if len(valid_opts) == 1:
            opt = valid_opts[0]
            strat = opt.get('triton_strategy', 'UNKNOWN')
            print(f"  ⚠ Parallelization: MUST use {opt['sequential_dim']}-seq, {opt['parallel_dim']}-par")
            print(f"  ⚠ Triton Strategy: {strat}")
        elif len(valid_opts) >= 2:
            print(f"  ✓ Parallelization: Multiple strategies valid")

    model_name = "claude-sonnet-4-20250514"
    message = client.messages.create(
        model=model_name,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )

    # Build complete response with metadata
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    war_info = "None" if not war_result or war_result['parallelization_safe'] else str(war_result['arrays_needing_copy'])

    par_info = "N/A"
    triton_strat_info = "N/A"
    if par_analysis:
        valid_opts = [opt for opt in par_analysis['options'] if opt['valid']]
        if len(valid_opts) == 1:
            opt = valid_opts[0]
            par_info = f"REQUIRED: {opt['sequential_dim']}-seq, {opt['parallel_dim']}-par ({opt['parallelism_type']})"
            triton_strat_info = opt.get('triton_strategy', 'UNKNOWN')
        elif len(valid_opts) >= 2:
            par_info = "Multiple valid"
            strats = [f"{opt['sequential_dim']}-seq:{opt.get('triton_strategy', 'UNKNOWN')}" for opt in valid_opts]
            triton_strat_info = ", ".join(strats)

    complete_response = f"""# LLM-Generated Triton Implementation for {kernel_name}
# Generated: {timestamp}
# Model: {model_name}
# Source: Original TSVC C code
# WAR Dependencies: {war_info}
# Parallelization: {par_info}
# Triton Strategy: {triton_strat_info}
# Stop Reason: {message.stop_reason}
# Input Tokens: {message.usage.input_tokens}
# Output Tokens: {message.usage.output_tokens}

{'='*80}
PROMPT:
{'='*80}
{prompt}

{'='*80}
RESPONSE:
{'='*80}
{message.content[0].text}
"""

    # Extract just the code content
    response_text = message.content[0].text
    triton_code = response_text
    if "```python" in triton_code:
        triton_code = triton_code.split("```python")[1].split("```")[0].strip()
    elif "```" in triton_code:
        triton_code = triton_code.split("```")[1].split("```")[0].strip()

    return triton_code, complete_response


def get_available_kernels() -> List[str]:
    """Get list of available TSVC kernels from the source file."""
    with open(TSVC_SOURCE, 'r') as f:
        content = f.read()

    # Match both s### functions (like s000, s111) and v functions (like va, vag, vsumr)
    s_kernels = re.findall(r'real_t (s\d+)\s*\(struct args_t', content)
    v_kernels = re.findall(r'real_t (v\w+)\s*\(struct args_t', content)
    all_kernels = s_kernels + v_kernels
    return sorted(set(all_kernels))


def main():
    llm_triton_dir = Path("llm_triton")
    llm_triton_dir.mkdir(exist_ok=True)
    raw_responses_dir = llm_triton_dir / "raw_responses"
    raw_responses_dir.mkdir(exist_ok=True)

    # Check for command-line argument
    if len(sys.argv) > 1:
        if sys.argv[1] == '--list':
            kernels = get_available_kernels()
            print(f"Available TSVC kernels ({len(kernels)}):")
            for k in kernels:
                print(f"  {k}")
            return
        elif sys.argv[1] == '--all':
            # Process ALL kernels from TSVC source (not filtered by PyTorch baselines)
            kernels = get_available_kernels()
            print(f"Processing {len(kernels)} kernels from TSVC source...")
        else:
            kernels = [sys.argv[1]]
            print(f"Processing single kernel: {kernels[0]}")
    else:
        print("Usage: python generate_llm_triton.py <kernel_name>")
        print("       python generate_llm_triton.py --all")
        print("       python generate_llm_triton.py --list")
        return

    success_count = 0
    fail_count = 0

    for kernel_name in kernels:
        try:
            triton_code, complete_response = generate_triton_from_tsvc(kernel_name)

            # Save with v3 suffix
            output_file = llm_triton_dir / f"{kernel_name}_triton_llm_v3.py"
            with open(output_file, 'w') as f:
                f.write(triton_code)

            raw_file = raw_responses_dir / f"{kernel_name}_raw_response_v3.txt"
            with open(raw_file, 'w') as f:
                f.write(complete_response)

            print(f"  ✓ Saved: {output_file}")
            success_count += 1

        except Exception as e:
            print(f"  ✗ Error for {kernel_name}: {e}")
            fail_count += 1

    print(f"\n✅ Complete: {success_count} success, {fail_count} failed")


if __name__ == "__main__":
    main()
