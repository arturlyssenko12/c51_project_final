#!/usr/bin/env python3
import sys

def detect_format(path):
    with open(path, 'rb') as f:
        header = f.read(512)

    # ZIP (including spanned/split‐ZIP parts)
    if header.startswith(b'\x50\x4B\x03\x04') \
    or header.startswith(b'\x50\x4B\x05\x06') \
    or header.startswith(b'\x50\x4B\x07\x08'):
        return 'ZIP archive (or ZIP split segment)'

    # Gzip (often .tar.gz)
    if header.startswith(b'\x1F\x8B\x08'):
        return 'GZIP compressed (e.g. .gz or .tar.gz chunk)'

    # Bzip2 (often .tar.bz2)
    if header.startswith(b'\x42\x5A\x68'):
        return 'BZIP2 compressed (e.g. .bz2 or .tar.bz2 chunk)'

    # 7-zip
    if header.startswith(b'\x37\x7A\xBC\xAF\x27\x1C'):
        return '7-Zip archive'

    # RAR
    if header.startswith(b'\x52\x61\x72\x21\x1A\x07\x00'):
        return 'RAR archive'

    # Plain TAR: at offset 257 the ustar magic appears
    if header[257:262] == b'ustar':
        return 'TAR archive'

    return 'Unknown format'

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]}  <filename>")
        sys.exit(1)

    fmt = detect_format(sys.argv[1])
    print(f"{sys.argv[1]}: {fmt}")