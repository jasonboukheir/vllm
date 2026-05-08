#!/usr/bin/env python3
"""Extract gated_delta_rule_kernel ELF section from _xpu_C.abi3.so.

Recreates the T23 disasm pipeline: dump OFFLOAD_DEVICE_CODE -> walk
concatenated ar archives -> pick ar11 (bf16/bf16 + ff instantiations)
-> read 64.bmg member as Xe2 ELF -> match section by mangled name
-> dump raw .text bytes for iga64.

Usage:
  extract_section.py <so_path> <out_dir> <pattern>
    pattern e.g. 'Lb1EEE' for IS_SPEC=true k=4 bf16/bf16
"""

from __future__ import annotations

import re
import struct
import subprocess
import sys
from pathlib import Path


def dump_offload(so: Path, out: Path) -> Path:
    blob = out / "OFFLOAD_DEVICE_CODE.raw"
    subprocess.run(
        [
            "objcopy",
            "--dump-section",
            f"OFFLOAD_DEVICE_CODE={blob}",
            str(so),
            "/dev/null",
        ],
        check=True,
    )
    return blob


def split_archives(blob: bytes) -> list[bytes]:
    magic = b"!<arch>\n"
    starts = [m.start() for m in re.finditer(re.escape(magic), blob)]
    archives: list[bytes] = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(blob)
        archives.append(blob[s:e])
    return archives


def read_ar_members(ar: bytes) -> list[tuple[str, bytes]]:
    pos = 8  # skip magic
    out: list[tuple[str, bytes]] = []
    while pos + 60 <= len(ar):
        hdr = ar[pos : pos + 60]
        if hdr[58:60] != b"`\n":
            break
        name = hdr[0:16].rstrip().decode("latin-1").rstrip("/")
        size = int(hdr[48:58].rstrip())
        pos += 60
        body = ar[pos : pos + size]
        out.append((name, body))
        pos += size + (size & 1)  # 2-byte align
    return out


def find_text_section(elf: bytes, pattern: str) -> tuple[str, int, int] | None:
    if elf[:4] != b"\x7fELF":
        return None
    # 64-bit ELF assumed (e_machine 0xCD = EM_INTELGT)
    e_shoff = struct.unpack_from("<Q", elf, 0x28)[0]
    e_shentsize = struct.unpack_from("<H", elf, 0x3A)[0]
    e_shnum = struct.unpack_from("<H", elf, 0x3C)[0]
    e_shstrndx = struct.unpack_from("<H", elf, 0x3E)[0]

    def shdr(i: int) -> tuple[int, int, int, int]:
        base = e_shoff + i * e_shentsize
        sh_name = struct.unpack_from("<I", elf, base + 0x00)[0]
        sh_offset = struct.unpack_from("<Q", elf, base + 0x18)[0]
        sh_size = struct.unpack_from("<Q", elf, base + 0x20)[0]
        return sh_name, sh_offset, sh_size, base

    _, str_off, str_sz, _ = shdr(e_shstrndx)
    strtab = elf[str_off : str_off + str_sz]

    def name_at(o: int) -> str:
        end = strtab.find(b"\x00", o)
        return strtab[o:end].decode("latin-1", errors="replace")

    for i in range(e_shnum):
        sh_name, sh_offset, sh_size, _ = shdr(i)
        nm = name_at(sh_name)
        if nm.startswith(".text._ZTSN3gdn23gated_delta_rule_kernelI") and pattern in nm:
            return nm, sh_offset, sh_size
    return None


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    so = Path(sys.argv[1])
    out = Path(sys.argv[2])
    pattern = sys.argv[3]
    out.mkdir(parents=True, exist_ok=True)

    blob_path = dump_offload(so, out)
    blob = blob_path.read_bytes()
    archives = split_archives(blob)
    print(f"# {len(archives)} archives in OFFLOAD_DEVICE_CODE", file=sys.stderr)

    # ar11 holds bf16/bf16 + ff. Walk all and pick by section match.
    matches: list[tuple[int, str, bytes]] = []
    for idx, ar in enumerate(archives):
        for name, body in read_ar_members(ar):
            if name != "64.bmg":
                continue
            hit = find_text_section(body, pattern)
            if hit is None:
                continue
            sec_name, off, size = hit
            matches.append((idx, sec_name, body[off : off + size]))
            print(f"# ar{idx:02d} {sec_name} size={size}", file=sys.stderr)

    if not matches:
        print(f"no section matching {pattern!r}", file=sys.stderr)
        return 1
    if len(matches) > 1:
        print(f"# WARNING: {len(matches)} matches; writing all", file=sys.stderr)

    for idx, sec_name, data in matches:
        suffix = re.sub(r"[^A-Za-z0-9]+", "_", sec_name)[:80]
        bin_path = out / f"ar{idx:02d}_{suffix}.bin"
        bin_path.write_bytes(data)
        print(str(bin_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
