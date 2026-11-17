"""Stage classifier annotations and, optionally, grouped policy inputs."""
import argparse
import gzip
import hashlib
import shutil
from pathlib import Path

from config import CLASSIFIER_WORK, POLICY_WORK, PROJECT_ROOT


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            sha.update(chunk)
    return sha.hexdigest()


def copy_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if digest(source) != digest(target):
            raise SystemExit(f'{target} already exists with different contents; use a new output directory.')
    else:
        shutil.copy2(source, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', action='store_true', help='Stage grouped action records and archived predictions.')
    args = parser.parse_args()
    source = PROJECT_ROOT / 'inputs/classifier'
    for name in ('human_labels.csv', 'training_rows.csv', 'model_manifest.json'):
        copy_file(source / name, CLASSIFIER_WORK / name)
    if args.policy:
        for path in (PROJECT_ROOT / 'inputs/policy').iterdir():
            if path.is_file():
                if path.suffix != '.gz':
                    copy_file(path, POLICY_WORK / 'policy_inputs/data' / path.name)
                    continue
                target = POLICY_WORK / 'policy_inputs/data' / path.stem
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(target.suffix + '.partial')
                try:
                    with gzip.open(path, 'rb') as source_file, temporary.open('wb') as output:
                        shutil.copyfileobj(source_file, output, 1024 * 1024)
                    if target.exists():
                        if digest(temporary) != digest(target):
                            raise SystemExit(f'{target} already exists with different contents; use a new output directory.')
                    else:
                        temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
    print('Classifier inputs staged in', CLASSIFIER_WORK)
    if args.policy:
        print('Policy inputs staged in', POLICY_WORK / 'policy_inputs/data')


if __name__ == '__main__':
    main()
