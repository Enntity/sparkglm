#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Stream a separate NVIDIA MTP expert overlay; stdlib only, never edit source."""
import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import pathlib
import re
import struct
import time

EXPERT = re.compile(r"^model\.language_model\.layers\.45\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
SHARDS = [f"model-{i:05}-of-00033.safetensors" for i in range(1, 4)]
CHUNK = 4 * 1024 * 1024

def header(path):
    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        if size > 64 * 1024 * 1024:
            raise ValueError("oversized header")
        return json.loads(f.read(size)), size + 8

def digest_span(f, start, size):
    f.seek(start)
    h = hashlib.sha256()
    while size:
        data = f.read(min(size, CHUNK))
        if not data:
            raise ValueError("truncated tensor")
        h.update(data)
        size -= len(data)
    return h.hexdigest()

def tensor_plan(h):
    result = {}
    offset = 0
    for name, info in h.items():
        if name == "__metadata__":
            continue
        dims = info["shape"]
        start, end = info["data_offsets"]
        entries = [(name, info["dtype"], dims, end-start)]
        if EXPERT.fullmatch(name):
            if info["dtype"] != "BF16" or len(dims) != 2:
                raise ValueError(f"not a BF16 matrix: {name}")
            n, k = dims
            if n <= 0 or k <= 0 or n % 16 or k % 16 or end-start != n*k*2:
                raise ValueError(f"invalid expert geometry: {name}")
            base = name[:-len(".weight")]
            entries = [(name, "U8", [n,k//2], n*k//2),
                       (base+".weight_scale", "F8_E4M3", [n,k//16], n*k//16),
                       (base+".weight_scale_2", "F32", [], 4)]
        for key, dtype, shape, size in entries:
            if key in result:
                raise ValueError(f"duplicate output key: {key}")
            result[key] = {"dtype":dtype, "shape":shape, "data_offsets":[offset,offset+size]}
            offset += size
    if "__metadata__" in h:
        result["__metadata__"] = h["__metadata__"]
    return result

class Quantizer:
    def __init__(self, path):
        self.lib = ctypes.CDLL(str(path.resolve()))
        self.lib.atlas_quantize.argtypes = [ctypes.c_void_p]*3 + [ctypes.POINTER(ctypes.c_float),ctypes.c_uint,ctypes.c_uint]
        self.lib.atlas_quantize.restype = ctypes.c_int
        self.lib.atlas_error.argtypes = [ctypes.c_int]
        self.lib.atlas_error.restype = ctypes.c_char_p
    def __call__(self, data, n, k):
        source = ctypes.create_string_buffer(data)
        packed = ctypes.create_string_buffer(n*k//2)
        scales = ctypes.create_string_buffer(n*k//16)
        scale2 = ctypes.c_float()
        code = self.lib.atlas_quantize(source, packed, scales, ctypes.byref(scale2), n, k)
        if code:
            raise RuntimeError(self.lib.atlas_error(code).decode())
        if not math.isfinite(scale2.value) or scale2.value <= 0:
            raise ValueError("invalid global scale")
        sf = scales.raw
        if any((v & 0x80) or v == 0x7f for v in sf):
            raise ValueError("negative or NaN E4M3 scale")
        return packed.raw, sf, struct.pack("<f",scale2.value)

def rewrite(source, dest, quantize):
    h, source_start = header(source)
    output = tensor_plan(h)
    encoded = json.dumps(output,separators=(",",":")).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    dest_start = 8 + len(encoded)
    receipts = {}
    with open(source,"rb") as src, open(dest,"xb") as dst:
        dst.write(struct.pack("<Q",len(encoded)))
        dst.write(encoded)
        for name, info in h.items():
            if name == "__metadata__":
                continue
            start,end = info["data_offsets"]
            src.seek(source_start+start)
            expected = output[name]["data_offsets"][0] + dest_start
            if dst.tell() != expected:
                raise ValueError("output offset disagreement")
            sha = hashlib.sha256()
            if EXPERT.fullmatch(name):
                data = src.read(end-start)
                if len(data) != end-start:
                    raise ValueError("truncated BF16 matrix")
                sha.update(data)
                packed, scales, scale2 = quantize(data,*info["shape"])
                n,k = info["shape"]
                if (len(packed),len(scales),len(scale2)) != (n*k//2,n*k//16,4):
                    raise ValueError("quantizer output lengths")
                scalar = struct.unpack("<f",scale2)[0]
                if not math.isfinite(scalar) or scalar <= 0:
                    raise ValueError("quantizer global scale")
                payload = packed+scales+scale2
                dst.write(payload)
                receipts[name] = {"source_sha256":sha.hexdigest(), "output_sha256":hashlib.sha256(payload).hexdigest(), "output_bytes":len(payload), "scale2":scalar, "converted":True}
            else:
                remaining=end-start
                while remaining:
                    data=src.read(min(remaining,CHUNK))
                    if not data:
                        raise ValueError("truncated source")
                    sha.update(data)
                    dst.write(data)
                    remaining-=len(data)
                receipts[name] = {"source_sha256":sha.hexdigest(), "output_sha256":sha.hexdigest(), "output_bytes":end-start, "converted":False}
    # Independently reopen and verify every payload, including every copied tensor.
    check, check_start = header(dest)
    if check != output or check_start != dest_start:
        raise ValueError("written header mismatch")
    with open(dest,"rb") as f:
        for name, receipt in receipts.items():
            start=output[name]["data_offsets"][0]
            if digest_span(f,dest_start+start,receipt["output_bytes"]) != receipt["output_sha256"]:
                raise ValueError(f"output checksum mismatch: {name}")
    return output, receipts

def verify_overlay(source, output):
    receipts=json.loads((output/"conversion.complete.json").read_text())
    checked=0
    for shard in SHARDS:
        original,_=header(source/shard)
        actual,start=header(output/shard)
        if actual != tensor_plan(original):
            raise ValueError(f"transferred metadata mismatch: {shard}")
        with open(output/shard,"rb") as f:
            for name,r in receipts["shards"][shard].items():
                offset=actual[name]["data_offsets"][0]
                if digest_span(f,start+offset,r["output_bytes"]) != r["output_sha256"]:
                    raise ValueError(f"transferred checksum mismatch: {name}")
                checked+=1
    print(json.dumps({"verified_payloads":checked,"overlay":str(output)}),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=pathlib.Path,required=True)
    p.add_argument("--output",type=pathlib.Path,required=True)
    p.add_argument("--library",type=pathlib.Path,default=pathlib.Path(__file__).with_name("libatlas_mtp_quantize.so"))
    p.add_argument("--verify-overlay",action="store_true",help="CPU-only verification of a completed or transferred overlay")
    p.add_argument("--one-matrix",action="store_true",help="GPU smoke test only; creates no overlay")
    args=p.parse_args()
    source=args.source.resolve()
    all_names=set()
    for shard in SHARDS:
        h,_=header(source/shard)
        tensor_plan(h)
        all_names.update(n for n in h if EXPERT.fullmatch(n))
    expected={f"model.language_model.layers.45.mlp.experts.{e}.{proj}.weight" for e in range(288) for proj in ("gate_proj","up_proj","down_proj")}
    if all_names != expected:
        raise ValueError("checkpoint does not contain exactly 864 expected MTP matrices")
    if args.verify_overlay:
        verify_overlay(source,args.output)
        return
    q=Quantizer(args.library)
    if args.one_matrix:
        name=sorted(all_names)[0]
        for shard in SHARDS:
            h,start=header(source/shard)
            if name in h:
                info=h[name]; lo,hi=info["data_offsets"]
                with open(source/shard,"rb") as f:
                    f.seek(start+lo); data=f.read(hi-lo)
                a=q(data,*info["shape"]); b=q(data,*info["shape"])
                if a != b:
                    raise ValueError("non-deterministic quantization")
                print(json.dumps({"name":name,"shape":info["shape"],"scale2":struct.unpack("<f",a[2])[0],"packed_sha256":hashlib.sha256(a[0]).hexdigest(),"repeat_equal":True}),flush=True)
                return
    output=args.output.absolute()
    output.mkdir(parents=True,exist_ok=False)
    index=json.loads((source/"model.safetensors.index.json").read_text())
    receipts={"source":str(source),"output":str(output),"started":time.time(),"shards":{}}
    for shard in SHARDS:
        print(f"Converting {shard}",flush=True)
        out, hashes=rewrite(source/shard,output/shard,q)
        receipts["shards"][shard]=hashes
        for name in out:
            if name != "__metadata__":
                index["weight_map"][name]=shard
        (output/"conversion.partial.json").write_text(json.dumps(receipts))
    total=0
    for shard in sorted(set(index["weight_map"].values())):
        h,_=header((output if shard in SHARDS else source)/shard)
        total+=sum(v["data_offsets"][1]-v["data_offsets"][0] for n,v in h.items() if n!="__metadata__")
    index.setdefault("metadata",{})["total_size"]=total
    (output/"model.safetensors.index.json").write_text(json.dumps(index))
    receipts["passthrough_links"]={}
    for path in source.iterdir():
        if path.name not in SHARDS and path.name!="model.safetensors.index.json" and path.is_file():
            target=output/path.name
            try:
                os.link(path.resolve(),target)
                mode="hardlink"
            except OSError as error:
                if error.errno not in (errno.EPERM,errno.EXDEV):
                    raise
                target.symlink_to(path.resolve())
                mode="absolute_symlink_requires_source_bind"
            receipts["passthrough_links"][path.name]={"mode":mode,"target":str(path.resolve())}
    receipts["finished"]=time.time()
    receipts["converted_matrices"]=len(all_names)
    receipts["copied_tensors_verified"]=sum(not r["converted"] for s in receipts["shards"].values() for r in s.values())
    receipts["total_size"]=total
    (output/"conversion.complete.json").write_text(json.dumps(receipts))
    (output/"conversion.partial.json").unlink()
    print(json.dumps({k:v for k,v in receipts.items() if k!="shards"}),flush=True)

if __name__=="__main__":
    main()
