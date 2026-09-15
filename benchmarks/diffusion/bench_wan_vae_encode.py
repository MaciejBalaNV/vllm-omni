# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark Cosmos3's Wan encoder against the reference on H100/GB200.

Examples::

    python benchmarks/diffusion/bench_wan_vae_encode.py --frames 1 --profile
    python benchmarks/diffusion/bench_wan_vae_encode.py --frames 93 --check-reconstruction --json results.json
    python benchmarks/diffusion/bench_wan_vae_encode.py --input pixels.pt --check-reconstruction
    torchrun --nproc-per-node 2 benchmarks/diffusion/bench_wan_vae_encode.py --vae-patch-parallel-size 2

``--input`` accepts preprocessed natural images/videos saved as a floating-point
RGB tensor [B, 3, T, H, W] in [-1, 1]. Input loading/transfers, JIT compilation,
quality checks and optional reference-decoder reconstruction are outside timing.
The default input is seeded uniform noise. The reference always runs first.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import statistics
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
import torch.distributed as dist

from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
    DistributedAutoencoderKLWan,
    OmniAutoencoderKLWan,
)
from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath import (
    VAE_FAST_PATH_LEVELS,
    install_wan_vae_encoder_fastpath,
)

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
TINY_CONFIG = dict(
    base_dim=20,
    decoder_base_dim=32,
    z_dim=48,
    dim_mult=[1, 2, 4, 4],
    num_res_blocks=2,
    temperal_downsample=[False, True, True],
    is_residual=True,
    patch_size=2,
    in_channels=12,
    out_channels=12,
    scale_factor_temporal=4,
    scale_factor_spatial=16,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="nvidia/Cosmos3-Nano")
    parser.add_argument("--subfolder", default="vae")
    parser.add_argument("--revision", default=None, help="Pin the HF checkpoint revision")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--size", default="1280x720", help="RGB input WxH")
    parser.add_argument("--frames", type=int, default=93)
    parser.add_argument("--input", type=Path, help="Preprocessed RGB .pt tensor; overrides size/frames")
    parser.add_argument("--dtype", choices=DTYPES, default="bf16")
    parser.add_argument("--fast-path", default=",".join(VAE_FAST_PATH_LEVELS))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tf32", choices=("on", "off"), default="on")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-rows", type=int, default=30)
    parser.add_argument("--check-reconstruction", action="store_true", help="Require reference-decode PSNR >= 50 dB")
    parser.add_argument("--json", type=Path, help="Write timings, environment and quality metrics")
    parser.add_argument("--tiling", action="store_true")
    parser.add_argument("--vae-patch-parallel-size", type=int, default=1)
    args = parser.parse_args()
    if args.warmup < 0 or args.iters < 1 or args.vae_patch_parallel_size < 1:
        parser.error("warmup must be nonnegative; iters and parallel size must be positive")
    levels = list(dict.fromkeys(args.fast_path.split(",")))
    if any(level not in VAE_FAST_PATH_LEVELS for level in levels):
        parser.error(f"fast-path levels must be in {VAE_FAST_PATH_LEVELS}")
    args.levels = ["off", *(level for level in levels if level != "off")]
    return args


def make_pixels(args):
    if args.input:
        pixels = torch.load(args.input, map_location="cpu", weights_only=True)
    else:
        width, height = map(int, args.size.lower().split("x"))
        generator = torch.Generator().manual_seed(args.seed)
        pixels = torch.rand((1, 3, args.frames, height, width), generator=generator) * 2 - 1
    if not isinstance(pixels, torch.Tensor) or pixels.ndim != 5 or pixels.shape[1] != 3:
        raise ValueError("input must be a [B, 3, T, H, W] RGB tensor")
    if not pixels.is_floating_point() or pixels.numel() == 0 or not torch.isfinite(pixels).all():
        raise ValueError("input must contain finite floating-point values")
    if pixels.min() < -1 or pixels.max() > 1:
        raise ValueError("input pixels must be normalized to [-1, 1]")
    if (pixels.shape[2] - 1) % 4 or pixels.shape[3] % 16 or pixels.shape[4] % 16:
        raise ValueError("Cosmos3 inputs require 1+4k frames and spatial dimensions divisible by 16")
    return pixels


def distributed_setup(args):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world != args.vae_patch_parallel_size:
        raise ValueError("torchrun world size must equal --vae-patch-parallel-size")
    if world == 1:
        torch.cuda.set_device(0)
        return 0
    from vllm_omni.diffusion.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    init_distributed_environment(world_size=world, rank=rank, local_rank=local_rank)
    initialize_model_parallel(sequence_parallel_size=world, ulysses_degree=world)
    return rank


def sync():
    torch.accelerator.synchronize()
    if dist.is_initialized():
        dist.barrier()


def max_across_ranks(value):
    tensor = torch.tensor(value, dtype=torch.float64, device="cuda")
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.item()


def load_vae(args):
    parallel = args.vae_patch_parallel_size > 1
    cls = DistributedAutoencoderKLWan if parallel else OmniAutoencoderKLWan
    if args.tiny:
        torch.manual_seed(args.seed)
        vae = cls(**TINY_CONFIG).to(device="cuda", dtype=DTYPES[args.dtype]).eval()
        if parallel:
            vae.init_distributed()
    else:
        vae = (
            cls.from_pretrained(
                args.model,
                subfolder=args.subfolder,
                revision=args.revision,
                torch_dtype=DTYPES[args.dtype],
            )
            .to("cuda")
            .eval()
        )
    if args.tiling or parallel:
        vae.enable_tiling()
    if parallel:
        vae.set_parallel_size(args.vae_patch_parallel_size, mode="tile")
    return vae


def differences(actual, reference):
    integer = torch.int16 if actual.element_size() == 2 else torch.int32
    equal = torch.equal(actual.contiguous().view(integer), reference.contiguous().view(integer))
    a, b = actual.float(), reference.float()
    error = a - b
    rmse = error.square().mean().sqrt().item()
    scale = max(b.square().mean().sqrt().item(), 1e-8)
    return {"bitwise_equal": equal, "max_abs_diff": error.abs().max().item(), "normalized_rmse": rmse / scale}


@torch.inference_mode()
def run_level(args, level, pixels, rank):
    vae = load_vae(args)
    inputs = pixels.to(device="cuda", dtype=DTYPES[args.dtype])
    sync()
    start = time.perf_counter()
    report = install_wan_vae_encoder_fastpath(vae, level=level)
    sync()
    install_time = max_across_ranks(time.perf_counter() - start)
    if level != "off" and not report.installed:
        raise RuntimeError(f"requested encoder fast path was not installed: {report.reason}")
    sync()
    start = time.perf_counter()
    vae.encode(inputs)
    sync()
    first_call = max_across_ranks(time.perf_counter() - start)
    for _ in range(args.warmup):
        vae.encode(inputs)
    sync()
    torch.accelerator.reset_peak_memory_stats()
    times = []
    posterior = None
    for _ in range(args.iters):
        posterior = None
        sync()
        start = time.perf_counter()
        posterior = vae.encode(inputs).latent_dist
        torch.accelerator.synchronize()
        times.append(max_across_ranks(time.perf_counter() - start))
    peak = max_across_ranks(torch.accelerator.max_memory_allocated()) / 2**30
    median = statistics.median(times)
    stats = dict(
        status="ok",
        level=level,
        install_s=install_time,
        first_call_s=first_call,
        median_s=median,
        min_s=min(times),
        times_s=times,
        peak_gib=peak,
        frames_per_s=inputs.shape[0] * inputs.shape[2] / median,
        patched=dict(report.patched),
        fused_silu=report.fused_silu_dtypes,
        checkpoint_revision=getattr(vae.config, "_commit_hash", None),
    )
    moments = posterior.parameters.detach().cpu()
    logvar = posterior.logvar.detach().cpu()
    mean = posterior.mode().detach().cpu()
    del posterior
    reconstruction = None
    if args.check_reconstruction:
        # No decoder fast path is installed in this benchmark. Every candidate
        # is reconstructed by the same reference decoder weights and settings.
        decoded = vae.decode(mean.to(device="cuda", dtype=DTYPES[args.dtype])).sample
        if rank == 0:
            reconstruction = decoded.cpu()
        del decoded
    if args.profile:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            vae.encode(inputs)
            sync()
        if rank == 0:
            kernels = {}
            for event in prof.events():
                if event.device_type.name == "CUDA":
                    entry = kernels.setdefault(event.name, {"name": event.name, "ms": 0.0, "calls": 0})
                    entry["ms"] += event.device_time_total / 1000
                    entry["calls"] += 1
            ordered = sorted(kernels.values(), key=lambda item: item["ms"], reverse=True)
            total = sum(item["ms"] for item in ordered)
            conv = sum(
                item["ms"]
                for item in ordered
                if any(tag in item["name"].lower() for tag in ("conv", "cudnn", "gemm", "cutlass", "xmma"))
            )
            stats["profile"] = {
                "kernel_time_ms": total,
                "conv_like_percent": 100 * conv / total if total else 0,
                "kernels": ordered[: args.profile_rows],
            }
            print(f"[{level}] CUDA operator profile")
            print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=args.profile_rows))
    del vae, inputs
    # Instance-bound forwards form cycles; collect them before the next level
    # so its peak memory excludes the previous model's parameters.
    gc.collect()
    torch.accelerator.empty_cache()
    return stats, moments, mean, logvar, reconstruction


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; use H100 or GB200 for target performance validation")
    rank = distributed_setup(args)
    torch.backends.cudnn.allow_tf32 = args.tf32 == "on"
    pixels = make_pixels(args)
    results = []
    reference = None
    failed = False
    environment = dict(
        gpu=torch.cuda.get_device_name(),
        capability=torch.cuda.get_device_capability(),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        diffusers=package_version("diffusers"),
        triton=package_version("triton"),
        vllm=package_version("vllm"),
        vllm_omni=package_version("vllm-omni"),
        platform=platform.platform(),
        model=args.model,
        revision=args.revision,
        dtype=args.dtype,
        tiny=args.tiny,
        tiling=args.tiling or args.vae_patch_parallel_size > 1,
        warmup=args.warmup,
        iters=args.iters,
        input_shape=list(pixels.shape),
        input=str(args.input) if args.input else "seeded_uniform",
        seed=args.seed,
        tf32=args.tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        deterministic=torch.are_deterministic_algorithms_enabled(),
        world_size=args.vae_patch_parallel_size,
    )
    try:
        for level in args.levels:
            try:
                result = run_level(args, level, pixels, rank)
            except torch.OutOfMemoryError:
                if dist.is_initialized():
                    raise  # A failed collective cannot be safely retried by one rank.
                results.append(dict(level=level, status="oom"))
                gc.collect()
                torch.accelerator.empty_cache()
                failed = True
                continue
            stats, moments, mean, logvar, reconstruction = result
            if level == "off":
                reference = result
            if reference is not None and rank == 0:
                stats["speedup"] = reference[0]["median_s"] / stats["median_s"]
                for name, actual, expected in (
                    ("moments", moments, reference[1]),
                    ("mean", mean, reference[2]),
                    ("logvar", logvar, reference[3]),
                ):
                    metrics = differences(actual, expected)
                    stats[name] = metrics
                    finite = math.isfinite(metrics["normalized_rmse"])
                    if level == "lossless":
                        failed |= not metrics["bitwise_equal"]
                    failed |= not finite or (level == "channels_last" and metrics["normalized_rmse"] > 0.01)
                if reconstruction is not None:
                    mse = (reconstruction.float() - reference[4].float()).square().mean().item()
                    psnr = math.inf if mse == 0 else 10 * math.log10(4 / mse)
                    stats["reconstruction_psnr_db"] = "inf" if psnr == math.inf else psnr
                    failed |= math.isnan(psnr) or psnr < 50
            results.append(stats)
            if rank == 0:
                print(json.dumps(stats, indent=2))
        if reference is None:
            failed = True
        if rank == 0 and args.json:
            args.json.write_text(json.dumps(dict(environment=environment, results=results), indent=2) + "\n")
        failed = bool(max_across_ranks(int(failed)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    if failed:
        raise SystemExit("Encoder validation failed or a workload ran out of memory; inspect the reported results")


if __name__ == "__main__":
    main()
