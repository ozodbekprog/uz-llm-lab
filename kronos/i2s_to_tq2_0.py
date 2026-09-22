#!/usr/bin/env python3
"""I2_S (bitnet.cpp fork) -> TQ2_0 (mainline llama.cpp) GGUF transkoder.

Formatlar (manbadan tekshirilgan):
  I2_S: 128 weight = 32 bayt; bayt gp: bits(6-2k) -> pozitsiya k*32+gp;
        kod = trit+1 (0=-1, 1=0, 2=+1); tensor oxirida float32 scale (n/4 da);
        tensor hajmi = n/4 + 32.
  TQ2_0: 256 weight = 64 bayt qs + f16 d (66 bayt); 32-baytli chunkda
        bayt m: bits(2n) -> pozitsiya n*32+m; qiymat = (kod-1)*d; d = blok amax.
  Tranzaksiya yo'q: tritlar aynan saqlanadi, faqat bit maydonlari joyi
  almashtiriladi va scale fp32->fp16 o'giriladi.

Ishga tushirish:
    .venv/bin/python kronos/i2s_to_tq2_0.py \
        --input bitnet/models/Falcon3-10B-Instruct-1.58bit/ggml-model-i2_s.gguf \
        --output /tmp/opencode/falcon10b-tq2_0.gguf
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

T_F32, T_F16, T_TQ2_0, T_I2_S = 0, 1, 35, 36
FTYPE_MOSTLY_TQ2_0 = 37
SCALAR_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


class Reader:
    def __init__(self, path: str):
        self.f = open(path, "rb")
        magic = self.f.read(4)
        if magic != b"GGUF":
            raise ValueError("GGUF magic emas")
        self.version, self.n_tensors, self.n_kv = struct.unpack("<IQQ", self.f.read(20))
        if self.version != 3:
            raise ValueError(f"GGUF versiya {self.version} qo'llab-quvvatlanmaydi")

    def _str(self) -> bytes:
        n = struct.unpack("<Q", self.f.read(8))[0]
        return self.f.read(n)

    def _str_raw(self) -> bytes:
        n = struct.unpack("<Q", self.f.read(8))[0]
        return struct.pack("<Q", n) + self.f.read(n)

    def read_kv(self):
        key = self._str().decode("utf-8")
        (vtype,) = struct.unpack("<I", self.f.read(4))
        head = struct.pack("<I", vtype)
        if vtype == 8:
            raw = self._str_raw()
            return key, vtype, head + raw
        if vtype == 9:
            (etype,) = struct.unpack("<I", self.f.read(4))
            (count,) = struct.unpack("<Q", self.f.read(8))
            buf = [etype.to_bytes(4, "little"), count.to_bytes(8, "little")]
            if etype == 8:
                for _ in range(count):
                    buf.append(self._str_raw())
            else:
                buf.append(self.f.read(count * SCALAR_SIZE[etype]))
            return key, vtype, head + b"".join(buf)
        raw = self.f.read(SCALAR_SIZE[vtype])
        return key, vtype, head + raw

    def read_tensor_info(self):
        name = self._str().decode("utf-8")
        (n_dims,) = struct.unpack("<I", self.f.read(4))
        dims = struct.unpack(f"<{n_dims}Q", self.f.read(8 * n_dims))
        ttype, offset = struct.unpack("<IQ", self.f.read(12))
        return {"name": name, "dims": dims, "type": ttype, "offset": offset,
                "ne": int(np.prod(dims))}


def i2s_to_tq2(data: np.ndarray, scale: float) -> np.ndarray:
    f0 = (data >> 6) & 3
    f1 = (data >> 4) & 3
    f2 = (data >> 2) & 3
    f3 = data & 3
    perm = (f0 | (f1 << 2) | (f2 << 4) | (f3 << 6)).astype(np.uint8)
    nb = perm.size // 64
    dst = np.empty((nb, 66), dtype=np.uint8)
    dst[:, :64] = perm.reshape(nb, 64)
    dst[:, 64:66] = np.frombuffer(np.array([np.float16(scale)], dtype=np.float16).tobytes(),
                                  dtype=np.uint8)
    return dst.reshape(-1)


def tensor_nbytes(ttype: int, ne: int) -> int:
    if ttype == T_I2_S:
        return ne // 4 + 32
    if ttype == T_TQ2_0:
        return (ne // 256) * 66
    if ttype == T_F16:
        return ne * 2
    if ttype == T_F32:
        return ne * 4
    raise ValueError(f"tur {ttype} qo'llab-quvvatlanmaydi")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    r = Reader(args.input)
    kvs = [r.read_kv() for _ in range(r.n_kv)]
    infos = [r.read_tensor_info() for _ in range(r.n_tensors)]
    align = 32
    for key, vtype, blob in kvs:
        if key == "general.alignment":
            align = struct.unpack("<I", blob[4:8])[0]

    header_end = r.f.tell()
    data_start = (header_end + align - 1) // align * align

    patched = []
    n_i2s = n_other = 0
    for key, vtype, blob in kvs:
        if key == "general.file_type":
            blob = blob[:4] + struct.pack("<I", FTYPE_MOSTLY_TQ2_0) + blob[8:]
        patched.append((key, vtype, blob))
        if key == "general.file_type":
            pass

    offset = 0
    for info in infos:
        if info["type"] == T_I2_S:
            if info["ne"] % 256:
                raise ValueError(f"{info['name']}: ne % 256 != 0")
            new_type = T_TQ2_0
            n_i2s += 1
        else:
            new_type = info["type"]
            n_other += 1
        info["new_type"] = new_type
        info["new_offset"] = offset
        offset = (offset + tensor_nbytes(new_type, info["ne"]) + align - 1) // align * align

    out = open(args.output, "wb")
    out.write(b"GGUF")
    out.write(struct.pack("<IQQ", r.version, r.n_tensors, r.n_kv))
    for key, vtype, blob in patched:
        kb = key.encode()
        out.write(struct.pack("<Q", len(kb)) + kb + blob)
    for info in infos:
        nb = info["name"].encode()
        out.write(struct.pack("<Q", len(nb)) + nb)
        out.write(struct.pack("<I", len(info["dims"])))
        out.write(struct.pack(f"<{len(info['dims'])}Q", *info["dims"]))
        out.write(struct.pack("<IQ", info["new_type"], info["new_offset"]))
    pos = out.tell()
    out.write(b"\x00" * ((data_start - pos) % align))

    done = 0
    for info in infos:
        want = data_start + info["new_offset"]
        cur = out.tell()
        if cur < want:
            out.write(b"\x00" * (want - cur))
        in_off = data_start + info["offset"]
        in_size = tensor_nbytes(info["type"], info["ne"])
        r.f.seek(in_off)
        raw = r.f.read(in_size)
        if info["type"] == T_I2_S:
            n4 = info["ne"] // 4
            scale = struct.unpack_from("<f", raw, n4)[0]
            codes = np.frombuffer(raw, dtype=np.uint8, count=n4)
            out.write(i2s_to_tq2(codes, scale).tobytes())
        else:
            out.write(raw)
        done += 1
        if done % 40 == 0 or done == len(infos):
            print(f"  [{done}/{len(infos)}] {info['name']}", flush=True)
    out.close()
    total = Path(args.output).stat().st_size / 1e9
    print(f"\nTayyor: {args.output} ({total:.2f} GB)")
    print(f"  I2_S->TQ2_0: {n_i2s} tensor; ko'chirildi: {n_other} tensor")


if __name__ == "__main__":
    sys.exit(main())
