"""
Unzip the corrupted-Zip64 DIOR archive.

'TGRSS ang DIOR remote sensing.zip' is a 6.6 GB Zip64 file whose EOCD locator
is missing AND whose per-entry Zip64 extras are garbled — the same double
pathology we hit on DOTA's part2.zip.  Python's stdlib zipfile and macOS
unzip both refuse it.  We use the same recovery strategy that worked for
DOTA: scan the entire data region for PK0304 (Local File Header)
signatures, match each one to a filename read from the (intact) central
directory, and let zlib stop at the end of each deflate stream on its own.

Layout of the outer zip:
  TGRSS ang DIOR remote sensing/
    DIOR-dataset/
      ImageSets.zip                 <- nested, contains train/val/test splits
      Annotations.zip               <- nested, contains VOC XMLs
      JPEGImages-test.zip           <- nested, redundant with loose files
      JPEGImages-test/*.jpg         <- 11,737 loose test images (IDs 11727-23463)

Note: JPEGImages-trainval (IDs 1-11725) are NOT in this archive; you'll
need a separate download for them before training.

Run:
    python unzip_dior.py
"""

import os
import struct
import time
import zlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ZIP_PATH = ROOT / "TGRSS ang DIOR remote sensing.zip"
DEST = ROOT / "DIOR_dataset"

# Pre-computed by inspection (see the diagnostic script)
CD_OFFSET = 7_064_481_831
CD_SIZE = 1_449_312

LFH_FMT = "<IHHHHHIIIHH"
LFH_SIZE = struct.calcsize(LFH_FMT)
CD_FMT = "<IHHHHHHIIIHHHHHII"
CD_ESIZE = struct.calcsize(CD_FMT)


def read_cd_names(f):
    """Parse the (intact) central directory into a name-set for name matching."""
    f.seek(CD_OFFSET)
    cd = f.read(CD_SIZE)
    names = set()
    p = 0
    while p < len(cd) and cd[p:p+4] == b"PK\x01\x02":
        fields = struct.unpack(CD_FMT, cd[p:p+CD_ESIZE])
        name_len, extra_len, comment_len = fields[10], fields[11], fields[12]
        name = cd[p+CD_ESIZE:p+CD_ESIZE+name_len].decode("utf-8", errors="replace")
        names.add(name)
        p += CD_ESIZE + name_len + extra_len + comment_len
    return names


def find_all_lfh_offsets(f):
    """Scan [0, CD_OFFSET) for every PK0304 signature."""
    SIG = b"PK\x03\x04"
    CHUNK = 16 * 1024 * 1024
    OVERLAP = 4
    positions = []
    pos = 0
    leftover = b""
    while pos < CD_OFFSET:
        read = min(CHUNK, CD_OFFSET - pos)
        f.seek(pos)
        buf = leftover + f.read(read)
        i = 0
        while True:
            j = buf.find(SIG, i)
            if j < 0:
                break
            abs_off = (pos - len(leftover)) + j
            if abs_off >= 0:
                positions.append(abs_off)
            i = j + 1
        leftover = buf[-OVERLAP:]
        pos += read
        if pos % (256 * 1024 * 1024) == 0 or pos >= CD_OFFSET:
            print(f"  scanned {pos/1024/1024/1024:.1f} GB, "
                  f"{len(positions):>7d} candidate LFHs so far", flush=True)
    return positions


def extract_outer(zip_path, dest):
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[outer] {zip_path.name}  ({zip_path.stat().st_size/1024/1024/1024:.2f} GB)")
    print(f"  -> {dest}")

    t0 = time.time()
    with open(zip_path, "rb") as f:
        cd_names = read_cd_names(f)
        print(f"  CD entries : {len(cd_names)}")
        offsets = find_all_lfh_offsets(f)
        print(f"  LFH signatures found : {len(offsets)} "
              f"(scan took {time.time()-t0:.1f}s)")

        extracted = 0
        skipped = 0
        t1 = time.time()
        for k, off in enumerate(offsets):
            f.seek(off)
            hdr = f.read(LFH_SIZE)
            if len(hdr) < LFH_SIZE:
                continue
            (lsig, lver, lflags, lmethod, lmtime, lmdate, lcrc,
             lcsize, lusize, lname_len, lextra_len) = struct.unpack(LFH_FMT, hdr)
            if lsig != 0x04034b50:
                continue
            if lname_len > 4096 or lextra_len > 8192:
                continue  # signature collision inside compressed data
            name_bytes = f.read(lname_len)
            try:
                name = name_bytes.decode("utf-8")
            except UnicodeDecodeError:
                continue
            if name not in cd_names:
                continue
            _ = f.read(lextra_len)

            if name.endswith("/"):
                (dest / name).mkdir(parents=True, exist_ok=True)
                continue

            data_start = f.tell()
            next_off = offsets[k + 1] if k + 1 < len(offsets) else CD_OFFSET
            data_len = next_off - data_start
            f.seek(data_start)
            data = f.read(data_len)
            try:
                if lmethod == 0:
                    n = lcsize if lcsize != 0xFFFFFFFF else min(data_len, lusize)
                    payload = data[:n]
                elif lmethod == 8:
                    payload = zlib.decompress(data, -15)
                else:
                    print(f"  !! unsupported method {lmethod} for {name}")
                    skipped += 1
                    continue
            except zlib.error as e:
                print(f"  !! decompress failed for {name}: {e}")
                skipped += 1
                continue

            out = dest / name
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "wb") as w:
                w.write(payload)
            extracted += 1
            if extracted % 500 == 0:
                print(f"  extracted {extracted:>7d} / ~{len(cd_names)}  "
                      f"({time.time()-t1:.1f}s)", flush=True)

    print(f"  done: {extracted} files extracted, {skipped} skipped, "
          f"total {time.time()-t0:.1f}s")


def extract_nested_zips(root):
    """After the outer extract, unpack the inner zips (Annotations, ImageSets)."""
    # The inner zips landed at:
    #   DIOR_dataset/TGRSS ang DIOR remote sensing/DIOR-dataset/*.zip
    inner_dir = root / "TGRSS ang DIOR remote sensing" / "DIOR-dataset"
    if not inner_dir.exists():
        print(f"  [warn] expected nested dir not found: {inner_dir}")
        return

    to_unpack = [
        ("Annotations.zip",     inner_dir / "Annotations"),
        ("ImageSets.zip",       inner_dir / "ImageSets"),
        ("JPEGImages-test.zip", inner_dir / "JPEGImages-test"),  # redundant, but harmless
    ]
    for zname, out_dir in to_unpack:
        zp = inner_dir / zname
        if not zp.exists():
            print(f"  [skip] {zname} not found")
            continue
        print(f"[inner] {zp.name} -> {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(zp) as zf:
                zf.extractall(out_dir)
            print(f"  done: {len(list(out_dir.rglob('*')))} items")
        except Exception as e:
            print(f"  !! standard extract failed for {zname}: {e}")
            print(f"     (nested zip may be corrupted too — inspect manually)")


def main():
    if not ZIP_PATH.exists():
        print(f"ERROR: {ZIP_PATH} not found")
        return
    extract_outer(ZIP_PATH, DEST)
    print()
    extract_nested_zips(DEST)
    print()
    print("=" * 64)
    print("  DIOR outer + nested extraction complete")
    print("=" * 64)
    print(f"  Output root : {DEST}")
    print()
    print("  What you have:")
    print("    ✓ 11,737 test JPEGs (IDs 11727-23463)")
    print("    ✓ Annotations (VOC XML for all 23,463 images)")
    print("    ✓ ImageSets (train.txt / val.txt / trainval.txt / test.txt)")
    print()
    print("  What is MISSING:")
    print("    ✗ JPEGImages-trainval (images with IDs 1-11725)")
    print("      → You need to obtain these separately before training.")


if __name__ == "__main__":
    main()
