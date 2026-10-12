"""Measure one fixed visual question, with independent full recomputation."""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import statistics
import time

import torch
from PIL import Image
from qwen_choice import QwenChoice


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    values = sorted(values)
    point = (len(values) - 1) * .95
    low = int(point)
    return {"n": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "min": values[0], "max": values[-1],
            "p95": values[low] + (values[min(low + 1, len(values)-1)] - values[low]) * (point-low)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results/run"))
    args = parser.parse_args()
    if args.rounds < 1 or args.warmup < 1:
        parser.error("Positive rounds and warmup are required")
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parent
    case_path = root / "examples/support_ticket.json"
    case = json.loads(case_path.read_text())
    image_path = case_path.parent / case["image"]
    image = Image.open(image_path).convert("RGB")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    torch.cuda.set_device(args.device)
    loading_start = time.perf_counter()
    runtime = QwenChoice(args.model, args.device, local_files_only=args.local_files_only)
    torch.cuda.synchronize()
    loading_s = time.perf_counter() - loading_start
    parameter_state = [(name, p.data_ptr(), p._version, p.requires_grad)
                       for name, p in runtime.model.named_parameters()]
    versions = {}
    for name in ("torch", "transformers", "triton", "flash-linear-attention",
                 "fla-core", "causal-conv1d", "Pillow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    config = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": "Qwen/Qwen3.5-4B", "python": platform.python_version(),
        "packages": versions, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(args.device), "dtype": "bfloat16",
        "loading_s": loading_s, "kernel_audit": runtime.kernel_audit,
        "image_size": list(image.size), "image_sha256": digest(image_path),
        "case_sha256": digest(case_path),
        "source_sha256": {name: digest(root / name) for name in
                          ("qwen_choice.py", "benchmark.py", "make_example.py")},
        "rounds": args.rounds, "warmup_per_profile": args.warmup,
        "question_count_per_request": 1, "image_count_per_request": 1,
        "batch_size": 1, "cross_request_cache": False,
        "image_processing": "Official defaults; no pixel override or manual downsampling",
        "request_timing": "In-memory RGB image to returned Python decision, synchronized CUDA boundaries; "
        "includes rendering, tokenization, image preprocessing, input hashing, H2D, model, and readout; "
        "excludes model loading, image file I/O, JSON log writes, cold/compile and warmup requests.",
        "model_gpu_timing": "CUDA events after H2D and before result D2H; includes any GPU idle time "
        "between host launches; not an isolated kernel profiler.",
        "scope": "One fictional example repeated for latency; no accuracy or probability-calibration claim.",
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    profiles = ["native", "eager", "accelerated"]
    records = []
    log = (args.output / "records.jsonl").open("w")

    def run(profile, phase, index):
        result = runtime.judge(image, case["question"], case["options"], profile=profile)
        result.update(phase=phase, index=index,
                      matches_fixture=result["choice"] == case["expected_choice"])
        log.write(json.dumps(result) + "\n")
        log.flush()
        records.append(result)
        print(json.dumps({k: result[k] for k in
                          ("phase", "profile", "index", "request_s", "choice", "valid")}), flush=True)
        return result

    for profile in profiles:
        run(profile, "cold", 0)
        for index in range(args.warmup):
            run(profile, "warmup", index)
    for index in range(args.rounds):
        order = profiles[index % len(profiles):] + profiles[:index % len(profiles)]
        for profile in order:
            run(profile, "measured", index)
    log.close()
    after = [(name, p.data_ptr(), p._version, p.requires_grad)
             for name, p in runtime.model.named_parameters()]
    if after != parameter_state:
        raise RuntimeError("Model parameters changed")
    measured = [r for r in records if r["phase"] == "measured"]
    summaries = {}
    for profile in profiles:
        rows = [r for r in measured if r["profile"] == profile]
        summaries[profile] = {
            **{key: stats([r[key] for r in rows]) for key in
               ("request_s", "model_gpu_s", "prepare_s")},
            "valid_count": sum(r["valid"] for r in rows),
            "fixture_match_count": sum(r["matches_fixture"] for r in rows),
            "choices": sorted(set(r["choice"] or "INVALID" for r in rows)),
            "generated_tokens": sorted(set(r["generated_tokens"] for r in rows)),
        }
    eager = [r for r in measured if r["profile"] == "eager"]
    fast = [r for r in measured if r["profile"] == "accelerated"]
    audit = {
        "parameters_unchanged": True,
        "all_inputs_identical": all(len({r["audit"][key] for r in records}) == 1
                                    for key in ("prompt_sha256", "input_ids_sha256", "pixel_values_sha256")),
        "one_visual_forward_every_request": all(r["audit"]["visual_forward_calls"] == 1 for r in records),
        "compiled_graph_ran_every_fast_request": all(r["audit"]["compiled_graph_calls"] > 0 for r in records
                                                       if r["profile"] == "accelerated"),
        "direct_one_forward_every_request": all(r["audit"]["model_forward_calls"] == 1 for r in records
                                                  if r["profile"] != "native"),
        "max_eager_vs_accelerated_probability_difference": max(
            abs(a["probabilities"][label] - b["probabilities"][label])
            for a, b in zip(eager, fast) for label in case["options"]),
        "all_outputs_valid": all(r["valid"] for r in records),
        "all_fixture_matches": all(r["matches_fixture"] for r in records),
    }
    summary = {"profiles": summaries, "verification": audit,
               "cold_request_s": {r["profile"]: r["request_s"] for r in records if r["phase"] == "cold"},
               "records_sha256": digest(args.output / "records.jsonl"),
               "config_sha256": digest(args.output / "config.json")}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if not all(value for key, value in audit.items() if key != "max_eager_vs_accelerated_probability_difference"):
        raise RuntimeError("Benchmark verification failed; inspect saved summary")


if __name__ == "__main__":
    main()
