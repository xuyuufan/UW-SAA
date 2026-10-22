"""Verify and reassemble the published sequential endpoint gzip parts.

The reconstructed gzip is written to the operating-system temporary directory by
default so it does not become an accidental oversized repository artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


DEFAULT_INDEX = Path(__file__).with_name("uwsaa_trajectory_endpoints.parts.json")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assemble_endpoint_gzip(index_path: Path = DEFAULT_INDEX, output_path: Path | None = None) -> Path:
    index_path = Path(index_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    expected_sha = index["payload_sha256"]
    expected_bytes = int(index["payload_bytes"])
    if output_path is None:
        output_path = Path(tempfile.gettempdir()) / f"uwsaa-trajectory-endpoints-{expected_sha}.csv.gz"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_file() and output_path.stat().st_size == expected_bytes and file_sha256(output_path) == expected_sha:
        return output_path

    temp_path = output_path.with_name(output_path.name + ".tmp")
    digest = hashlib.sha256()
    total = 0
    try:
        with temp_path.open("wb") as destination:
            for part in index["parts"]:
                name = part["filename"]
                if Path(name).name != name:
                    raise ValueError(f"Invalid part filename in index: {name!r}")
                source = index_path.parent / name
                if not source.is_file():
                    raise FileNotFoundError(source)
                if source.stat().st_size != int(part["bytes"]):
                    raise ValueError(f"Size mismatch for {name}")
                if file_sha256(source) != part["sha256"]:
                    raise ValueError(f"SHA-256 mismatch for {name}")
                with source.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        destination.write(block)
                        digest.update(block)
                        total += len(block)
        if total != expected_bytes:
            raise ValueError(f"Reassembled byte count mismatch: expected {expected_bytes}, got {total}")
        if digest.hexdigest() != expected_sha:
            raise ValueError("Reassembled payload SHA-256 mismatch")
        os.replace(temp_path, output_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX, help="parts index JSON")
    parser.add_argument("--output", type=Path, default=None, help="output gzip path (default: OS temporary directory)")
    args = parser.parse_args()
    result = assemble_endpoint_gzip(args.index, args.output)
    print(result)


if __name__ == "__main__":
    main()
