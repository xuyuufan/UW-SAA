"""Recover non-GP endpoints from the frozen pre-migration Git tree."""
from __future__ import annotations
import argparse, collections, csv, gzip, hashlib, io, json, subprocess, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_COMMIT = "a39d0ac0e454f48d591d573868afa21531db34e7"
INDEX_PATH = "reproducibility/data/uwsaa_trajectory_endpoints.parts.json"
INDEX_SHA256 = "401f05465fd034a7ad0732eb89cb0c481311e016825c727cded9b6c0b7e29275"
OUTPUT_STEM = "retained_non_gp_endpoints.csv.gz"
MAX_PART_BYTES = 40_000_000
DROP_SOLVERS = {"GP", "MEAN6"}

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def git_show(relative_path, destination):
    with destination.open("wb") as output:
        subprocess.run(["git", "show", f"{SOURCE_COMMIT}:{relative_path}"],
                       cwd=ROOT, stdout=output, check=True)

def fetch_source(temp_root):
    index_file = temp_root / "source.parts.json"
    git_show(INDEX_PATH, index_file)
    if sha256(index_file) != INDEX_SHA256:
        raise ValueError("Pinned legacy parts index SHA-256 mismatch")
    index = json.loads(index_file.read_text(encoding="utf-8"))
    payload = temp_root / "source.csv.gz"
    digest, total, parts = hashlib.sha256(), 0, []
    with payload.open("wb") as output:
        for item in index["parts"]:
            name = str(item["filename"])
            if Path(name).name != name:
                raise ValueError(f"Invalid part filename: {name!r}")
            part_path = temp_root / name
            git_show(f"reproducibility/data/{name}", part_path)
            actual_bytes, actual_sha = part_path.stat().st_size, sha256(part_path)
            if actual_bytes != int(item["bytes"]) or actual_sha != item["sha256"]:
                raise ValueError(f"Pinned source part failed verification: {name}")
            with part_path.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(block); digest.update(block); total += len(block)
            parts.append({"path": f"reproducibility/data/{name}",
                          "bytes": actual_bytes, "sha256": actual_sha})
            part_path.unlink()
    if total != int(index["payload_bytes"]) or digest.hexdigest() != index["payload_sha256"]:
        raise ValueError("Reassembled pinned endpoint source failed verification")
    return index, payload, parts

def split_payload(payload, output_dir):
    total = payload.stat().st_size
    count = max(1, (total + MAX_PART_BYTES - 1) // MAX_PART_BYTES)
    result = []
    with payload.open("rb") as source:
        for number in range(count):
            name = f"{OUTPUT_STEM}.part-{number + 1:03d}-of-{count:03d}"
            destination, digest, size = output_dir / name, hashlib.sha256(), 0
            with destination.open("xb") as target:
                remaining = MAX_PART_BYTES
                while remaining:
                    block = source.read(min(1024 * 1024, remaining))
                    if not block: break
                    target.write(block); digest.update(block); size += len(block); remaining -= len(block)
            if not 0 < size <= MAX_PART_BYTES:
                raise ValueError(f"Invalid output part size for {name}: {size}")
            result.append({"filename": name, "bytes": size, "sha256": digest.hexdigest()})
    if sum(row["bytes"] for row in result) != total:
        raise ValueError("Filtered gzip split is incomplete")
    return result

def recover(output_dir):
    output_dir = output_dir.resolve()
    if output_dir == ROOT or ROOT in output_dir.parents:
        raise ValueError("output directory must be outside the checkout")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="uwsaa-legacy-endpoints-", dir=output_dir) as temp_name:
        temp_root = Path(temp_name)
        index, source_payload, source_parts = fetch_source(temp_root)
        filtered_payload = temp_root / "retained_non_gp.csv.gz"
        solver_counts, removed_counts = collections.Counter(), collections.Counter()
        source_status_counts, retained_status_counts, coverage_counts = collections.Counter(), collections.Counter(), collections.Counter()
        source_rows = retained_rows = removed_rows = 0
        with gzip.open(source_payload, "rb") as compressed:
            source_text = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
            reader = csv.DictReader(source_text)
            if not reader.fieldnames or "solver" not in reader.fieldnames:
                raise ValueError("Pinned endpoint CSV has no solver column")
            fields = list(reader.fieldnames)
            coverage_fields = [n for n in ("study_group", "study_membership", "evidence_status")
                               if n in fields]
            with filtered_payload.open("wb") as raw_output:
                gzip_output = gzip.GzipFile(filename="", mode="wb", fileobj=raw_output, mtime=0)
                output_text = io.TextIOWrapper(gzip_output, encoding="utf-8", newline="")
                writer = csv.DictWriter(output_text, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                for row in reader:
                    source_rows += 1
                    solver = str(row.get("solver", "")); solver_counts[solver] += 1
                    status = str(row.get("evidence_status", ""))
                    source_status_counts[status] += 1
                    coverage = "|".join(str(row.get(n, "")) for n in coverage_fields)
                    if solver.upper() in DROP_SOLVERS:
                        removed_counts[solver] += 1; removed_rows += 1; continue
                    writer.writerow(row); retained_rows += 1
                    retained_status_counts[status] += 1
                    coverage_counts[coverage] += 1
                output_text.close()
        if source_rows != 1_722_662:
            raise ValueError(f"Expected 1,722,662 source rows; found {source_rows}")
        if retained_rows + removed_rows != source_rows:
            raise ValueError("Source row accounting does not reconcile")
        if sum(retained_status_counts.values()) != retained_rows:
            raise ValueError("Retained evidence-status counts do not reconcile")
            raise ValueError("Source row accounting does not reconcile")
        output_parts = split_payload(filtered_payload, output_dir)
        payload_sha = sha256(filtered_payload)
        parts_index = {
            "schema_version": 1, "payload": OUTPUT_STEM,
            "payload_bytes": filtered_payload.stat().st_size,
            "payload_sha256": payload_sha,
            "part_bytes_target": MAX_PART_BYTES,
            "parts": [{"sequence": i + 1, **part} for i, part in enumerate(output_parts)],
        }
        parts_index_path = output_dir / "retained_non_gp_endpoints.parts.json"
        with parts_index_path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(parts_index, stream, indent=2, ensure_ascii=False); stream.write("\n")
        receipt = {
            "schema": "uwsaa_non_gp_endpoint_recovery_v1",
            "source_commit": SOURCE_COMMIT,
            "source_index_path": INDEX_PATH,
            "source_index_sha256": INDEX_SHA256,
            "source_payload_path": index["payload"],
            "source_payload_bytes": source_payload.stat().st_size,
            "source_payload_sha256": sha256(source_payload),
            "source_rows": source_rows,
            "source_parts": source_parts,
            "source_evidence_status_counts": dict(sorted(source_status_counts.items())),
            "filter": "drop rows where solver.upper() is GP or MEAN6",
            "removed_row_count": removed_rows,
            "removed_solver_counts": dict(sorted(removed_counts.items())),
            "retained_row_count": retained_rows,
            "retained_solver_counts": dict(sorted((n,c) for n,c in solver_counts.items()
                                                  if n.upper() not in DROP_SOLVERS)),
            "retained_evidence_status_counts": dict(sorted(retained_status_counts.items())),
            "study_coverage_fields": coverage_fields,
            "retained_study_coverage_counts": dict(sorted(coverage_counts.items())),
            "numeric_text_and_provenance": "CSV values remain strings; numeric text, provenance, and evidence_status fields are preserved.",
            "filtered_payload_bytes": filtered_payload.stat().st_size,
            "filtered_payload_sha256": payload_sha,
            "parts_index": "retained_non_gp_endpoints.parts.json",
            "parts_index_sha256": sha256(parts_index_path),
            "part_bytes_max": MAX_PART_BYTES,
            "parts": output_parts,
        }
        with (output_dir / "retained_non_gp_endpoints.receipt.json").open(
                "x", encoding="utf-8", newline="\n") as stream:
            json.dump(receipt, stream, indent=2, ensure_ascii=False); stream.write("\n")
        return receipt

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="empty directory outside the checkout for filtered data and receipt")
    args = parser.parse_args()
    receipt = recover(args.output_dir)
    keys = ("source_commit", "source_rows", "removed_row_count", "removed_solver_counts",
            "retained_row_count", "retained_solver_counts", "filtered_payload_bytes",
            "filtered_payload_sha256", "parts")
    summary = {key: receipt[key] for key in keys}
    summary.update(status="RECOVERED_NON_GP_ONLY",
                   receipt=str(args.output_dir.resolve() / "retained_non_gp_endpoints.receipt.json"))
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
