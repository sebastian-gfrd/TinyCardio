"""
Verification of PhysioNet Datasets Integrity
============================================
Checks presence and SHA-256 checksums for records in:
  - data/nsrdb
  - data/edb
  - data/vfdb
  - data/sddb
"""

import os
import hashlib
import sys

DATASETS = {
    'nsrdb': 'data/nsrdb',
    'edb': 'data/edb',
    'vfdb': 'data/vfdb',
    'sddb': 'data/sddb'
}


def verify_sha256(db_name: str, folder_path: str):
    sha_file = os.path.join(folder_path, 'SHA256SUMS.txt')
    if not os.path.exists(sha_file):
        print(f"[{db_name.upper()}] Warning: SHA256SUMS.txt not found in {folder_path}")
        return

    passed, failed, missing = 0, 0, 0

    with open(sha_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            expected_hash = parts[0]
            filename = parts[1].lstrip('*')
            filepath = os.path.join(folder_path, filename)

            if not os.path.exists(filepath):
                # Deprecated or old backup files (.hea-, .atr-) may not have been downloaded
                missing += 1
                continue

            h = hashlib.sha256()
            with open(filepath, 'rb') as bf:
                while chunk := bf.read(65536):
                    h.update(chunk)
            actual_hash = h.hexdigest()

            if actual_hash == expected_hash:
                passed += 1
            else:
                failed += 1
                print(f"[{db_name.upper()}] MISMATCH: {filename}")

    status = "OK" if failed == 0 else "FAIL"
    print(f"[{db_name.upper():5s}] Status: {status} | Passed: {passed:3d} | Missing (backups/legacy): {missing:2d} | Corrupted: {failed:1d}")


def verify_records_completeness(db_name: str, folder_path: str):
    rec_file = os.path.join(folder_path, 'RECORDS')
    if not os.path.exists(rec_file):
        print(f"[{db_name.upper()}] Warning: RECORDS file not found.")
        return

    with open(rec_file, 'r') as f:
        records = [l.strip() for l in f if l.strip()]

    incomplete = []
    for r in records:
        hea = os.path.exists(os.path.join(folder_path, f"{r}.hea"))
        dat = os.path.exists(os.path.join(folder_path, f"{r}.dat"))
        ann = (os.path.exists(os.path.join(folder_path, f"{r}.atr")) or
               os.path.exists(os.path.join(folder_path, f"{r}.ari")))
        if not (hea and dat and ann):
            incomplete.append(r)

    if not incomplete:
        print(f"[{db_name.upper():5s}] All {len(records)} records have complete .hea, .dat and annotation files.")
    else:
        print(f"[{db_name.upper():5s}] Incomplete records: {incomplete}")


if __name__ == '__main__':
    print("=" * 65)
    print("PHYSIOPNET DATASET INTEGRITY & COMPLETENESS VERIFIER")
    print("=" * 65)
    for name, path in DATASETS.items():
        verify_records_completeness(name, path)
        verify_sha256(name, path)
        print("-" * 65)
