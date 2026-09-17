#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from pipeflow_ext.data import (
    dataset_fingerprint, dataset_record, generate_dataset, load_dataset, save_dataset,
)


OOD = ["high_flow", "high_roughness", "extreme_diameter_ratio", "combined"]


def specifications(train_n: int, val_n: int, test_n: int) -> list[dict]:
    specs = [
        {"name": "b2_iid_train.npz", "branches": 2, "regime": "iid", "split": "train", "samples": train_n, "seed": 42},
        {"name": "b2_iid_val.npz", "branches": 2, "regime": "iid", "split": "val", "samples": val_n, "seed": 43},
        {"name": "b2_iid_test.npz", "branches": 2, "regime": "iid", "split": "test", "samples": test_n, "seed": 44},
    ]
    for offset, regime in enumerate(OOD):
        specs.append({"name": f"b2_{regime}_test.npz", "branches": 2, "regime": regime,
                      "split": "test", "samples": test_n, "seed": 144 + offset})
    for branches in [3, 5]:
        specs.extend([
            {"name": f"b{branches}_iid_train.npz", "branches": branches, "regime": "iid",
             "split": "train", "samples": train_n, "seed": 40 + branches},
            {"name": f"b{branches}_iid_val.npz", "branches": branches, "regime": "iid",
             "split": "val", "samples": val_n, "seed": 50 + branches},
            {"name": f"b{branches}_iid_test.npz", "branches": branches, "regime": "iid",
             "split": "test", "samples": test_n, "seed": 60 + branches},
        ])
    return specs


def expected_config(args) -> dict:
    return {
        "protocol": "cai-colebrook-canonical-v1",
        "train_n": args.train_n,
        "val_n": args.val_n,
        "test_n": args.test_n,
        "min_re": args.min_re,
        "specifications": specifications(args.train_n, args.val_n, args.test_n),
    }


def verify_record(root: Path, expected: dict, record: dict) -> list[str]:
    path = root / expected["name"]
    if not path.is_file():
        return [f"missing file: {path}"]
    current = dataset_record(path, root)
    errors = []
    for key in ["dataset_id", "samples", "branches", "regime", "seed"]:
        if current[key] != record.get(key):
            errors.append(f"{expected['name']} {key}: {current[key]} != {record.get(key)}")
    return errors


def verify_manifest(root: Path, manifest: dict, config: dict | None = None) -> None:
    errors = []
    if config is not None and manifest.get("config") != config:
        errors.append("requested configuration differs from the frozen manifest")
    records = manifest.get("datasets", {})
    specs = manifest.get("config", {}).get("specifications", [])
    for spec in specs:
        if spec["name"] not in records:
            errors.append(f"manifest record missing: {spec['name']}")
            continue
        errors.extend(verify_record(root, spec, records[spec["name"]]))
    suite_material = json.dumps(
        {"config": manifest.get("config"), "datasets": records},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    calculated_suite_id = hashlib.sha256(suite_material).hexdigest()
    if calculated_suite_id != manifest.get("suite_id"):
        errors.append("suite_id does not match manifest content")
    if errors:
        raise SystemExit("Canonical dataset verification failed:\n- " + "\n- ".join(errors))


def prepare(args) -> None:
    root = Path(args.root)
    manifest_path = root / "dataset_manifest.json"
    config = expected_config(args)
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        verify_manifest(root, manifest, config)
        print(json.dumps({"status": "verified", "suite_id": manifest["suite_id"],
                          "manifest": str(manifest_path)}, indent=2))
        return
    root.mkdir(parents=True, exist_ok=True)
    records = {}
    for spec in config["specifications"]:
        path = root / spec["name"]
        if path.is_file():
            current = dataset_record(path, root)
            metadata = {key: current[key] for key in ["samples", "branches", "regime", "seed"]}
            expected = {key: spec[key] for key in ["samples", "branches", "regime", "seed"]}
            if metadata != expected:
                raise SystemExit(
                    f"Unfrozen existing dataset has unexpected metadata: {path}\n"
                    f"actual={metadata}\nexpected={expected}\nUse a new DATA_ROOT.")
            print(f"[adopt] {path}")
        else:
            print(f"[generate] {path}")
            data = generate_dataset(spec["samples"], spec["branches"], spec["regime"],
                                    spec["seed"], args.min_re)
            save_dataset(path, data)
        records[spec["name"]] = dataset_record(path, root)
    suite_material = json.dumps(
        {"config": config, "datasets": records}, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    manifest = {
        "suite_id": hashlib.sha256(suite_material).hexdigest(),
        "config": config,
        "datasets": records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    verify_manifest(root, manifest, config)
    print(json.dumps({"status": "created", "suite_id": manifest["suite_id"],
                      "manifest": str(manifest_path)}, indent=2))


def verify(args) -> None:
    root = Path(args.root)
    path = root / "dataset_manifest.json"
    if not path.is_file():
        raise SystemExit(f"Missing frozen manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    verify_manifest(root, manifest)
    print(manifest["suite_id"] if args.print_suite_id else
          json.dumps({"valid": True, "suite_id": manifest["suite_id"]}, indent=2))


def verify_result(args) -> None:
    data_id = dataset_fingerprint(load_dataset(args.data))
    with Path(args.result).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"Empty result file: {args.result}")
    errors = []
    for index, row in enumerate(rows, 1):
        if row.get("dataset_id") != data_id:
            errors.append(f"row {index}: dataset_id mismatch")
        if row.get("suite_id") != args.suite_id:
            errors.append(f"row {index}: suite_id mismatch")
        if row.get("checkpoint") != args.checkpoint:
            errors.append(f"row {index}: checkpoint mismatch")
    if errors:
        raise SystemExit("Result verification failed:\n- " + "\n- ".join(errors))
    print(json.dumps({"valid": True, "result": args.result, "dataset_id": data_id}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create and enforce one immutable experiment dataset suite")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--root", default="data_canonical_v1")
    p.add_argument("--train-n", type=int, default=20000)
    p.add_argument("--val-n", type=int, default=4000)
    p.add_argument("--test-n", type=int, default=4000)
    p.add_argument("--min-re", type=float, default=4000.0)
    p.set_defaults(func=prepare)
    p = sub.add_parser("verify")
    p.add_argument("--root", default="data_canonical_v1")
    p.add_argument("--print-suite-id", action="store_true")
    p.set_defaults(func=verify)
    p = sub.add_parser("verify-result")
    p.add_argument("--result", required=True); p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True); p.add_argument("--suite-id", required=True)
    p.set_defaults(func=verify_result)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
