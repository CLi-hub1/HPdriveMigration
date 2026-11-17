"""Check the SHA-256 manifest for all packaged files."""
import csv
import hashlib
from pathlib import Path

from config import PROJECT_ROOT


def main():
    manifest = PROJECT_ROOT / 'SHA256SUMS.csv'
    failures = []
    with manifest.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        path = PROJECT_ROOT / row['file']
        if not path.is_file():
            failures.append(f"Missing: {row['file']}")
            continue
        digest = hashlib.sha256()
        with path.open('rb') as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(chunk)
        if path.stat().st_size != int(row['bytes']) or digest.hexdigest() != row['sha256']:
            failures.append(f"Mismatch: {row['file']}")
    if failures:
        raise SystemExit('\n'.join(failures))
    print(f'PASS: {len(rows)} packaged files match SHA256SUMS.csv')


if __name__ == '__main__':
    main()
