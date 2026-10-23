"""Verify and extract the public Mendeley v3 AMCTS author instances."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = Path(__file__).resolve().parent
DEFAULT_ARCHIVE = TASK_DIR / "data" / "mendeley_3j52cjrm52_v3_downloadall.zip"
DEFAULT_INDEX = TASK_DIR / "data" / "instances_metadata_index.json"
DEFAULT_OUT = TASK_DIR / "data" / "verified_instances"
DEFAULT_RECEIPT = TASK_DIR / "data" / "dataset_audit.json"
INSTANCE_PREFIX = "Probabilistic Container Relocation Problem/Data/Instances/"
GLOBAL_RPM_MEMBER = "Probabilistic Container Relocation Problem/Data/retrieval_probability.json"
EXPECTED_CASES = 1680
EXPECTED_GLOBAL_RPM_SHA256 = "2ca3a32a5f98586db37e4a961f99f88128c62652b6132d9dd4f9b44d50c877f9"
EXPECTED_GLOBAL_RPM_SIZE = 525983
EXPECTED_EXAMPLE_SHA256 = "1b90a00f3b6f5a62f84732a07fa531d3b025c645c80e3a692789159d2542b7e6"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate_case(filename: str, raw_bytes: bytes) -> dict[str, object]:
    data = json.loads(raw_bytes)
    if not isinstance(data, dict):
        raise ValueError(f"{filename}: expected top-level object")
    required = {"initial_layout", "max_tier", "retrieval_sequence", "retrieval_probability"}
    if not required.issubset(data):
        raise ValueError(f"{filename}: missing fields {sorted(required - set(data))}")

    layout = data["initial_layout"]
    if not isinstance(layout, list) or not layout or any(not isinstance(stack, list) for stack in layout):
        raise ValueError(f"{filename}: invalid initial_layout")
    flat: list[int] = []
    for stack in layout:
        for item in stack:
            if isinstance(item, bool) or not isinstance(item, int):
                raise ValueError(f"{filename}: non-integer container ID")
            flat.append(item)
    n = len(flat)
    if n == 0 or sorted(flat) != list(range(n)):
        raise ValueError(f"{filename}: IDs must be unique and contiguous from 0")

    max_tier = data["max_tier"]
    if isinstance(max_tier, bool) or not isinstance(max_tier, int) or max_tier <= 0:
        raise ValueError(f"{filename}: invalid max_tier")
    if any(len(stack) > max_tier for stack in layout):
        raise ValueError(f"{filename}: stack exceeds max_tier")

    sequence = data["retrieval_sequence"]
    if (not isinstance(sequence, list) or len(sequence) != n
            or any(isinstance(item, bool) or not isinstance(item, int) for item in sequence)
            or sorted(sequence) != list(range(n))):
        raise ValueError(f"{filename}: retrieval_sequence must be a permutation of IDs")

    rpm = data["retrieval_probability"]
    if not isinstance(rpm, list) or len(rpm) != n:
        raise ValueError(f"{filename}: RPM row count must equal container count")
    widths: set[int] = set()
    for row in rpm:
        if not isinstance(row, list) or not row:
            raise ValueError(f"{filename}: invalid RPM row")
        values = [float(value) for value in row]
        widths.add(len(values))
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError(f"{filename}: RPM has invalid probability")
        if not math.isclose(math.fsum(values), 1.0, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError(f"{filename}: RPM row does not sum to one")
    if len(widths) != 1:
        raise ValueError(f"{filename}: inconsistent RPM period count")
    if widths != {12}:
        raise ValueError(f"{filename}: expected 12 author retrieval periods, got {sorted(widths)}")

    stem_parts = PurePosixPath(filename).stem.split("-")
    if len(stem_parts) != 4 or not all(part.isdigit() for part in stem_parts):
        raise ValueError(f"{filename}: unexpected case key format")
    width, height, declared_n, _seed = (int(part) for part in stem_parts)
    if width != len(layout) or height != max_tier or declared_n != n:
        raise ValueError(
            f"{filename}: filename geometry {width}-{height}-{declared_n} disagrees with "
            f"JSON layout={len(layout)}, max_tier={max_tier}, N={n}"
        )
    config = "-".join(stem_parts[:3])
    return {
        "filename": filename,
        "sha256": _sha256(raw_bytes),
        "size": len(raw_bytes),
        "container_count": n,
        "max_tier": max_tier,
        "stack_count": len(layout),
        "period_count": next(iter(widths)),
        "configuration": config,
        "seed": int(stem_parts[3]),
    }


def audit(archive: Path, index_path: Path, output_dir: Path, receipt_path: Path) -> dict[str, object]:
    index_bytes = index_path.read_bytes()
    index_sha = _sha256(index_bytes)
    index = json.loads(index_bytes)
    if not isinstance(index, list):
        raise ValueError("metadata index must be a JSON list")
    expected = {item["filename"]: item for item in index}
    if len(expected) != len(index) or len(expected) != EXPECTED_CASES:
        raise ValueError(f"metadata index must contain {EXPECTED_CASES} unique cases")

    archive_bytes = archive.read_bytes()
    archive_sha = _sha256(archive_bytes)
    with zipfile.ZipFile(archive) as zf:
        members = [item.filename for item in zf.infolist() if not item.is_dir()]
        if len(members) != len(set(members)):
            raise ValueError("zip contains duplicate member paths")
        instance_paths: dict[str, str] = {}
        examples: list[str] = []
        global_rpm: list[str] = []
        for member in members:
            normalized = member.replace("\\", "/")
            if normalized.endswith("/Instances/example.json"):
                examples.append(member)
            elif normalized.startswith(INSTANCE_PREFIX):
                basename = PurePosixPath(normalized).name
                if basename in instance_paths:
                    raise ValueError(f"duplicate instance basename: {basename}")
                instance_paths[basename] = member
            elif normalized == GLOBAL_RPM_MEMBER:
                global_rpm.append(member)
            else:
                raise ValueError(f"unexpected non-directory zip member: {member}")

        actual_names = set(instance_paths)
        missing = sorted(set(expected) - actual_names)
        extra = sorted(actual_names - set(expected))
        if missing or extra or len(actual_names) != EXPECTED_CASES:
            raise ValueError(f"instance member mismatch missing={missing[:3]} extra={extra[:3]} count={len(actual_names)}")
        if len(examples) != 1 or len(global_rpm) != 1:
            raise ValueError(f"expected one example and one global RPM file; got {len(examples)}, {len(global_rpm)}")

        rows: list[dict[str, object]] = []
        configs: Counter[str] = Counter()
        ns: Counter[int] = Counter()
        for filename in sorted(expected):
            raw = zf.read(instance_paths[filename])
            metadata = expected[filename]
            digest = _sha256(raw)
            expected_digest = metadata["content_details"]["sha256_hash"]
            expected_size = int(metadata["content_details"]["size"])
            if digest != expected_digest or len(raw) != expected_size:
                raise ValueError(f"metadata byte/hash mismatch for {filename}")
            record = _validate_case(filename, raw)
            rows.append(record)
            configs[str(record["configuration"])] += 1
            ns[int(record["container_count"])] += 1

        if len(configs) != 56 or any(count != 30 for count in configs.values()):
            raise ValueError(f"expected 56 configurations × 30 cases, got {len(configs)} configs; counts={sorted(set(configs.values()))}")
        seeds_by_config: dict[str, set[int]] = {}
        for row in rows:
            seeds_by_config.setdefault(str(row["configuration"]), set()).add(int(row["seed"]))
        if any(seeds != set(range(30)) for seeds in seeds_by_config.values()):
            raise ValueError("each configuration must contain exactly seeds 0 through 29")

        example_sha = _sha256(zf.read(examples[0]))
        if example_sha != EXPECTED_EXAMPLE_SHA256:
            raise ValueError("example.json SHA-256 mismatch")
        global_rpm_bytes = zf.read(global_rpm[0])
        global_rpm_sha = _sha256(global_rpm_bytes)
        if global_rpm_sha != EXPECTED_GLOBAL_RPM_SHA256 or len(global_rpm_bytes) != EXPECTED_GLOBAL_RPM_SIZE:
            raise ValueError("known non-scoring global retrieval_probability.json hash/size mismatch")
        # Extract only after every metadata, hash, and schema check passes.
        output_dir.mkdir(parents=True, exist_ok=True)
        for filename, member in instance_paths.items():
            (output_dir / filename).write_bytes(zf.read(member))

    receipt = {
        "schema": "mendeley_author_dataset_audit_v1",
        "status": "PASS_BYTES_HASH_AND_SCHEMA",
        "dataset_version": 3,
        "dataset_page": "https://data.mendeley.com/datasets/3j52cjrm52/3",
        "download_all_url": "https://data.mendeley.com/public-api/zip/3j52cjrm52/download/3",
        "archive": str(archive.resolve()),
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": archive_sha,
        "metadata_index": str(index_path.resolve()),
        "metadata_index_sha256": index_sha,
        "instance_count": len(rows),
        "configuration_count": len(configs),
        "cases_per_configuration": 30,
        "container_count_distribution": dict(sorted(ns.items())),
        "instance_hashes_match_metadata": True,
        "schema_validation": "all 1680 files: layout/IDs/capacity/RPM/sequence",
        "example_member": examples[0],
        "example_sha256_not_scored": example_sha,
        "global_rpm_member": global_rpm[0],
        "global_rpm_sha256_not_used_by_policy": global_rpm_sha,
        "global_rpm_size_bytes_not_used_by_policy": len(global_rpm_bytes),
        "extracted_instances_dir": str(output_dir.resolve()),
        "files": rows,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--metadata-index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args()
    receipt = audit(args.archive, args.metadata_index, args.output_dir, args.receipt)
    print(json.dumps({key: value for key, value in receipt.items() if key != "files"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
