import os

with open("src/kernels/metal/head.rs", "r") as f:
    lines = f.readlines()

def find_line(pattern):
    for i, line in enumerate(lines):
        if pattern in line:
            return i
    return -1

drop_start = find_line("impl Drop for MetalGraph {")

# head.rs only needs encode_output_head
with open("src/kernels/metal/head.rs", "w") as f:
    for line in lines[:drop_start]:
        f.write(line)
        if line.strip() == "return ok;":
            f.write("    }\n}\n")
            break

# We already have drop in graph.rs, let's make sure.
