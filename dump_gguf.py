import struct
import json
import sys

def read_u64(f): return struct.unpack('<Q', f.read(8))[0]
def read_u32(f): return struct.unpack('<I', f.read(4))[0]
def read_string(f):
    l = read_u64(f)
    return f.read(l).decode('utf-8')

with open("ds4flash.gguf", "rb") as f:
    magic = f.read(4)
    if magic != b'GGUF':
        print("Not GGUF")
        sys.exit(1)
    version = read_u32(f)
    n_tensors = read_u64(f)
    n_kv = read_u64(f)
    print(f"Version: {version}, Tensors: {n_tensors}, KV: {n_kv}")
    
    # skip kvs
    for _ in range(n_kv):
        read_string(f)
        vtype = read_u32(f)
        if vtype == 8: # string
            read_string(f)
        elif vtype == 9: # array
            atype = read_u32(f)
            alen = read_u64(f)
            if atype == 8:
                for _ in range(alen): read_string(f)
            elif atype in (4, 5, 6): # 32-bit
                f.read(4 * alen)
            elif atype in (10, 11):
                f.read(8 * alen)
            elif atype == 7:
                f.read(alen)
            else:
                print(f"Unknown array type {atype}")
                sys.exit(1)
        elif vtype in (4, 5): f.read(4) # i32/u32
        elif vtype in (6,): f.read(4) # f32
        elif vtype in (7,): f.read(1) # bool
        elif vtype in (10, 11): f.read(8) # u64/i64
        else: print(f"Unknown KV type {vtype}"); break
        
    for _ in range(n_tensors):
        name = read_string(f)
        ndim = read_u32(f)
        dims = [read_u64(f) for _ in range(ndim)]
        ttype = read_u32(f)
        offset = read_u64(f)
        if "token_embd" in name:
            print(f"Tensor {name}: dims={dims}, type={ttype}, offset={offset}")