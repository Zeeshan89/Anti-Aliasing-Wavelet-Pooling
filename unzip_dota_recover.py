"""Recover the remaining images from part2.zip by scanning local file headers.

The archive's central-directory extras are vendor-garbled, so we can't trust
LFH offsets in the CD. We scan the whole data region for every PK0304
signature, match each one to a filename from the CD, and extract any image
that's still missing on disk. zlib stops at the end of the deflate stream
on its own, so we can hand it a generous buffer and ignore csize entirely.
"""

import os
import struct
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PART2 = ROOT / "DOTA_1_5" / "Train_dota" / "part2.zip"
DEST = ROOT / "DOTA_1_5" / "Train_dota" / "images"
CD_OFFSET = 4753685876
CD_SIZE = 47145

LFH_FMT = "<IHHHHHIIIHH"
LFH_SIZE = struct.calcsize(LFH_FMT)
CD_FMT = "<IHHHHHHIIIHHHHHII"
CD_ESIZE = struct.calcsize(CD_FMT)


def list_cd_names(f):
    f.seek(CD_OFFSET)
    cd = f.read(CD_SIZE)
    names = []
    p = 0
    while p < len(cd) and cd[p:p+4] == b"PK\x01\x02":
        fields = struct.unpack(CD_FMT, cd[p:p+CD_ESIZE])
        name_len, extra_len, comment_len = fields[10], fields[11], fields[12]
        name = cd[p+CD_ESIZE:p+CD_ESIZE+name_len].decode("utf-8", errors="replace")
        names.append(name)
        p += CD_ESIZE + name_len + extra_len + comment_len
    return set(names)


def find_all_lfh(f):
    """Find every PK0304 offset within [0, CD_OFFSET)."""
    SIG = b"PK\x03\x04"
    CHUNK = 8 * 1024 * 1024
    OVERLAP = 4
    positions = []
    pos = 0
    f.seek(0)
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
    return positions


def main():
    present = set(os.listdir(DEST))
    print(f"already present: {len(present)} files")

    with open(PART2, "rb") as f:
        cd_names = list_cd_names(f)
        print(f"CD entry names: {len(cd_names)}")

        t0 = time.time()
        offsets = find_all_lfh(f)
        print(f"scanned all LFH signatures: {len(offsets)} candidates in {time.time()-t0:.1f}s")

        extracted = 0
        for off in offsets:
            f.seek(off)
            hdr = f.read(LFH_SIZE)
            if len(hdr) < LFH_SIZE:
                continue
            (lsig, lver, lflags, lmethod, lmtime, lmdate, lcrc,
             lcsize, lusize, lname_len, lextra_len) = struct.unpack(LFH_FMT, hdr)
            if lsig != 0x04034b50:
                continue
            if lname_len > 1024 or lextra_len > 4096:
                continue  # signature collision in compressed data
            name_bytes = f.read(lname_len)
            try:
                name = name_bytes.decode("utf-8")
            except UnicodeDecodeError:
                continue
            if name not in cd_names:
                continue
            _ = f.read(lextra_len)  # skip extras
            basename = os.path.basename(name)
            if not name.startswith("images/") or not basename or basename in present:
                continue

            data_start = f.tell()
            # Read up to next LFH (or CD start) — zlib stops at end of stream.
            idx = offsets.index(off)
            end = offsets[idx + 1] if idx + 1 < len(offsets) else CD_OFFSET
            data_len = end - data_start
            f.seek(data_start)
            data = f.read(data_len)

            try:
                if lmethod == 0:
                    # stored: trust lcsize if it's not the Zip64 marker
                    n = lcsize if lcsize != 0xFFFFFFFF else min(data_len, lusize)
                    payload = data[:n]
                elif lmethod == 8:
                    payload = zlib.decompress(data, -15)
                else:
                    print(f"  ! unsupported method {lmethod} for {name}")
                    continue
            except zlib.error as e:
                print(f"  ! decompress failed for {name}: {e}")
                continue

            (DEST / basename).write_bytes(payload)
            present.add(basename)
            extracted += 1
            print(f"  + {name} ({len(payload)} bytes)", flush=True)

    print(f"extracted {extracted} new files; images/ now has {len(os.listdir(DEST))} files")


if __name__ == "__main__":
    main()
