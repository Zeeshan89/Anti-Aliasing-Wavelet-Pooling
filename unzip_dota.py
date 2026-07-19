"""Unzip DOTA 1.5 train/val archives.

part2.zip from the official DOTA mirror is missing the Zip64 EOCD locator/record,
so Python's stdlib `zipfile` and macOS `unzip` both refuse it even though the
data + central directory are intact. We extract that one manually by parsing
the central directory at its known offset; the rest go through `zipfile`.
"""

import os
import struct
import sys
import time
import zipfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRAIN = ROOT / "DOTA_1_5" / "Train_dota"
VAL = ROOT / "DOTA_1_5" / "Val_dota"


def extract_standard(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[zipfile] {zip_path.name} -> {dest}")
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.infolist()
        total = len(members)
        t0 = time.time()
        for i, m in enumerate(members, 1):
            zf.extract(m, dest)
            if i % 50 == 0 or i == total:
                print(f"  {i}/{total} ({time.time()-t0:.1f}s)", flush=True)
    print(f"  done: {total} entries in {time.time()-t0:.1f}s")


CD_ENTRY_FMT = "<IHHHHHHIIIHHHHHII"
CD_ENTRY_SIZE = struct.calcsize(CD_ENTRY_FMT)
LFH_FMT = "<IHHHHHIIIHH"
LFH_SIZE = struct.calcsize(LFH_FMT)


def extract_part2_manual(zip_path: Path, dest: Path, cd_offset: int, cd_size: int) -> None:
    """Manually walk the central directory of a Zip64 file whose EOCD locator is missing."""
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[manual ] {zip_path.name} -> {dest}  (cd_offset={cd_offset}, cd_size={cd_size})")
    t0 = time.time()
    count = 0
    with open(zip_path, "rb") as f:
        f.seek(cd_offset)
        cd_bytes = f.read(cd_size)
        # Parse central directory entries
        entries = []
        p = 0
        while p < len(cd_bytes):
            if cd_bytes[p:p+4] != b"PK\x01\x02":
                # End of CD entries
                break
            (sig, ver_made, ver_need, flags, method, mtime, mdate,
             crc, csize, usize, name_len, extra_len, comment_len,
             disk_start, int_attr, ext_attr, lfh_offset
             ) = struct.unpack(CD_ENTRY_FMT, cd_bytes[p:p+CD_ENTRY_SIZE])
            name = cd_bytes[p+CD_ENTRY_SIZE : p+CD_ENTRY_SIZE+name_len].decode("utf-8", errors="replace")
            # Handle Zip64 extra field
            extra = cd_bytes[p+CD_ENTRY_SIZE+name_len : p+CD_ENTRY_SIZE+name_len+extra_len]
            real_usize, real_csize, real_lfh = usize, csize, lfh_offset
            ep = 0
            while ep + 4 <= len(extra):
                tag, esize = struct.unpack("<HH", extra[ep:ep+4])
                if tag == 0x0001:  # Zip64 extra
                    pos = ep + 4
                    if usize == 0xFFFFFFFF:
                        real_usize = struct.unpack("<Q", extra[pos:pos+8])[0]; pos += 8
                    if csize == 0xFFFFFFFF:
                        real_csize = struct.unpack("<Q", extra[pos:pos+8])[0]; pos += 8
                    if lfh_offset == 0xFFFFFFFF:
                        real_lfh = struct.unpack("<Q", extra[pos:pos+8])[0]; pos += 8
                ep += 4 + esize
            entries.append((name, method, real_csize, real_usize, real_lfh, crc))
            p += CD_ENTRY_SIZE + name_len + extra_len + comment_len

        total = len(entries)
        print(f"  parsed {total} CD entries", flush=True)

        for i, (name, method, csize, usize, lfh, crc) in enumerate(entries, 1):
            out_path = dest / name
            if name.endswith("/") or usize == 0 and csize == 0 and name.endswith("/"):
                out_path.mkdir(parents=True, exist_ok=True)
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)

            f.seek(lfh)
            lfh_bytes = f.read(LFH_SIZE)
            (lsig, lver, lflags, lmethod, lmtime, lmdate, lcrc,
             lcsize, lusize, lname_len, lextra_len
             ) = struct.unpack(LFH_FMT, lfh_bytes)
            assert lsig == 0x04034b50, f"bad LFH sig at {lfh} for {name}"
            f.seek(lfh + LFH_SIZE + lname_len + lextra_len)
            data = f.read(csize)

            if method == 0:
                payload = data
            elif method == 8:
                payload = zlib.decompress(data, -15)
            else:
                raise RuntimeError(f"Unsupported compression method {method} for {name}")

            with open(out_path, "wb") as out:
                out.write(payload)
            count += 1
            if i % 50 == 0 or i == total:
                print(f"  {i}/{total} ({time.time()-t0:.1f}s)", flush=True)
    print(f"  done: {count} files in {time.time()-t0:.1f}s")


def main():
    jobs = [
        # (zip path, destination, mode, optional kwargs)
        (TRAIN / "part1.zip",          TRAIN,                          "standard"),
        (TRAIN / "part3.zip",          TRAIN,                          "standard"),
        (TRAIN / "labelTxt.zip",       TRAIN / "labelTxt",             "standard"),
        (TRAIN / "Train_Task2_gt.zip", TRAIN,                          "standard"),
        (VAL   / "part1.zip",          VAL,                            "standard"),
        (VAL   / "labelTxt.zip",       VAL / "labelTxt",               "standard"),
        (VAL   / "Val_Task2_gt.zip",   VAL,                            "standard"),
        (VAL   / "DOTA-v1.5_val.zip",     VAL / "DOTA-v1.5_val",       "standard"),
        (VAL   / "DOTA-v1.5_val_hbb.zip", VAL / "DOTA-v1.5_val_hbb",   "standard"),
        # part2.zip: broken Zip64 EOCD, extract manually
        (TRAIN / "part2.zip",          TRAIN,                          "manual"),
    ]

    for entry in jobs:
        zip_path, dest, mode = entry[0], entry[1], entry[2]
        if not zip_path.exists():
            print(f"!! missing {zip_path}", file=sys.stderr)
            continue
        if mode == "standard":
            extract_standard(zip_path, dest)
        elif mode == "manual":
            # Pre-computed from inspection: first PK0102 at 4753685876, EOCD at 4753733021
            extract_part2_manual(zip_path, dest, cd_offset=4753685876, cd_size=47145)
        print()


if __name__ == "__main__":
    main()
