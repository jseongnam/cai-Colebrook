from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import (
    audit_legacy_npz, dataset_fingerprint, generate_dataset, load_dataset,
    save_dataset,
)


def cmd_generate(args):
    data = generate_dataset(args.samples, args.branches, args.regime, args.seed, args.min_re)
    save_dataset(args.output, data)
    print(json.dumps({"output": args.output, "samples": args.samples,
                      "branches": args.branches, "regime": args.regime}, indent=2))


def cmd_audit(args):
    reports = [audit_legacy_npz(path) for path in args.paths]
    text = json.dumps(reports, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    if any(r["unsafe"] for r in reports):
        raise SystemExit(2)


def _torch_commands(args):
    import torch
    from .experiment import (
        ModelConfig, TrainConfig, benchmark_runtime, evaluate_checkpoint,
        save_rows, train_model,
    )
    if args.command == "train":
        train_cfg = TrainConfig(epochs=args.epochs, patience=args.patience,
                                batch_size=args.batch_size, learning_rate=args.lr,
                                weight_decay=args.weight_decay, device=args.device,
                                num_workers=args.num_workers)
        model_cfg = ModelConfig(branch_hidden=args.branch_hidden,
                                global_hidden=args.global_hidden,
                                context_hidden=args.context_hidden,
                                depth=args.depth, dropout=args.dropout)
        path = train_model(load_dataset(args.train), load_dataset(args.val), args.input_mode,
                           args.seed, args.output, train_cfg, model_cfg,
                           args.target_mode, args.flow_parameterization, args.suite_id)
        print(path)
    elif args.command == "verify-checkpoint":
        meta = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        expected = {
            "suite_id": args.suite_id,
            "train_dataset_id": dataset_fingerprint(load_dataset(args.train)),
            "val_dataset_id": dataset_fingerprint(load_dataset(args.val)),
            "input_mode": args.input_mode,
            "target_mode": args.target_mode,
            "flow_parameterization": args.flow_parameterization,
            "seed": args.seed,
        }
        mismatches = {key: {"checkpoint": meta.get(key), "expected": value}
                      for key, value in expected.items() if meta.get(key) != value}
        if mismatches:
            print(json.dumps({"valid": False, "mismatches": mismatches}, indent=2))
            raise SystemExit(3)
        print(json.dumps({"valid": True, **expected}, indent=2))
    elif args.command == "evaluate":
        data = load_dataset(args.data)
        meta = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        experiment_variant = Path(args.checkpoint).parent.parent.name
        dataset_id = dataset_fingerprint(data)
        rows = evaluate_checkpoint(args.checkpoint, data, args.device, args.tol, args.max_iter)
        for row in rows:
            row.update({"dataset": args.data, "checkpoint": args.checkpoint,
                        "input_mode": meta["input_mode"], "seed": meta["seed"],
                        "target_mode": meta.get("target_mode", "correction"),
                        "flow_parameterization": meta.get("flow_parameterization", "logit"),
                        "experiment_variant": experiment_variant,
                        "suite_id": meta.get("suite_id", "unregistered"),
                        "dataset_id": dataset_id,
                        "train_dataset_id": meta.get("train_dataset_id", "unknown"),
                        "val_dataset_id": meta.get("val_dataset_id", "unknown"),
                        "branches": int(data["branches"]), "regime": str(data["regime"])})
        save_rows(args.output, rows)
        print(json.dumps(rows, indent=2))
    elif args.command == "runtime":
        data = load_dataset(args.data)
        meta = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        experiment_variant = Path(args.checkpoint).parent.parent.name
        dataset_id = dataset_fingerprint(data)
        result = benchmark_runtime(
            args.checkpoint, data, args.device, args.samples, args.warmup,
            args.repeats, args.tol, args.max_iter, args.inference_batch_size)
        result.update({"dataset": args.data, "checkpoint": args.checkpoint,
                       "input_mode": meta["input_mode"], "seed": meta["seed"],
                       "target_mode": meta.get("target_mode", "correction"),
                       "flow_parameterization": meta.get("flow_parameterization", "logit"),
                       "experiment_variant": experiment_variant,
                       "suite_id": meta.get("suite_id", "unregistered"),
                       "dataset_id": dataset_id,
                       "train_dataset_id": meta.get("train_dataset_id", "unknown"),
                       "val_dataset_id": meta.get("val_dataset_id", "unknown"),
                       "branches": int(data["branches"]), "regime": str(data["regime"])})
        save_rows(args.output, [result])
        print(json.dumps(result, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description="Leakage-safe extended pipe-flow experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("generate")
    p.add_argument("--samples", type=int, required=True)
    p.add_argument("--branches", type=int, default=2)
    p.add_argument("--regime", choices=["iid", "high_flow", "high_roughness",
                                        "extreme_diameter_ratio", "combined"], default="iid")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-re", type=float, default=4000.0)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("audit-legacy")
    p.add_argument("paths", nargs="+")
    p.add_argument("--output")
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("train")
    p.add_argument("--train", required=True); p.add_argument("--val", required=True)
    p.add_argument("--input-mode", choices=["full", "no_state", "physics_only"], required=True)
    p.add_argument("--target-mode", choices=["correction", "direct"], default="correction")
    p.add_argument("--flow-parameterization", choices=["logit", "raw_ratio"], default="logit")
    p.add_argument("--suite-id", default="unregistered")
    p.add_argument("--seed", type=int, default=42); p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu"); p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=35); p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-4); p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--branch-hidden", type=int, default=128); p.add_argument("--global-hidden", type=int, default=64)
    p.add_argument("--context-hidden", type=int, default=128); p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.05)
    p.set_defaults(func=_torch_commands)

    p = sub.add_parser("verify-checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--train", required=True); p.add_argument("--val", required=True)
    p.add_argument("--suite-id", required=True)
    p.add_argument("--input-mode", choices=["full", "no_state", "physics_only"], required=True)
    p.add_argument("--target-mode", choices=["correction", "direct"], required=True)
    p.add_argument("--flow-parameterization", choices=["logit", "raw_ratio"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.set_defaults(func=_torch_commands)

    for name in ["evaluate", "runtime"]:
        p = sub.add_parser(name)
        p.add_argument("--checkpoint", required=True); p.add_argument("--data", required=True)
        p.add_argument("--device", default="cpu"); p.add_argument("--output", required=True)
        p.add_argument("--tol", type=float, default=1e-10); p.add_argument("--max-iter", type=int, default=30)
        if name == "runtime":
            p.add_argument("--samples", type=int, default=500); p.add_argument("--warmup", type=int, default=2)
            p.add_argument("--repeats", type=int, default=10)
            p.add_argument("--inference-batch-size", type=int, default=0,
                           help="Neural forward-pass chunk size; 1 measures online latency, 0 uses all samples")
        p.set_defaults(func=_torch_commands)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
