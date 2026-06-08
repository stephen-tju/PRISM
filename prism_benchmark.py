import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass

from tqdm.asyncio import tqdm

from LLMServe.prism import (
    PeakWorkloadAnticipator,
    PrismScaler,
    load_prism_config,
)
from LLMServe.prism.scheduler import PrismScheduler


logger = logging.getLogger("LLMServe.prism.benchmark")
scheduler = None


@dataclass
class PrismBenchmarkConfig:
    request_config: dict
    profile_config: dict
    instance_config: dict
    scheduler_config: dict
    prism_config: object


def build_config(args):
    """Build benchmark, serving, and PRISM configuration dictionaries."""

    if args.request_num < 1:
        raise ValueError("request_num must be at least 1")
    if args.num_instances < 1:
        raise ValueError("num_instances must be at least 1")
    if args.qps <= 0:
        raise ValueError("qps must be positive")

    if args.load in ["ShareGPT", "LMSYS-Chat-1M"]:
        load_mode = "prompt_dataset"
    elif args.load in ["BurstGPT", "Azure_code", "Azure_conv"]:
        load_mode = "workload_trace_sample"
    elif args.load == "workload_trace":
        load_mode = "workload_trace"
    else:
        load_mode = "random_generated"

    workload_mode = "trace" if args.workload in [
        "BurstGPT",
        "Azure_code",
        "Azure_code_peak",
        "Azure_conv",
        "Azure_conv_peak",
    ] else "random_generated"

    request_config = {
        "request_num": args.request_num,
        "benchmark_duration": args.benchmark_duration,
        "load": args.load,
        "load_dataset_path": args.load_dataset_path,
        "seed": args.seed,
        "random_prompt_lens_mean": args.random_prompt_lens_mean,
        "random_prompt_lens_range": args.random_prompt_lens_range,
        "random_response_lens_mean": args.random_response_lens_mean,
        "random_response_lens_range": args.random_response_lens_range,
        "mixing_propotion": args.mixing_propotion,
        "workload": args.workload,
        "workload_trace_path": args.workload_trace_path,
        "workload_timescale": args.workload_timescale,
        "workload_sampling_interval": args.workload_sampling_interval,
        "workload_trace_range_L": args.workload_trace_range_L,
        "workload_trace_range_R": args.workload_trace_range_R,
        "qps": args.qps,
        "coefficient_variation": args.coefficient_variation,
        "load_mode": load_mode,
        "workload_mode": workload_mode,
        "dataset": args.load,
    }

    if load_mode == "prompt_dataset" and args.load_dataset_path is None:
        request_config["load_dataset_path"] = os.path.join(
            "../data/datasets/", os.path.join(args.load, "cleaned.csv")
        )
    if workload_mode == "trace" and args.workload_trace_path is None:
        request_config["workload_trace_path"] = os.path.join(
            "../data/workloads/", os.path.join(args.workload, args.workload + ".csv")
        )

    result_dir = args.result_dir
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    case_path = next_case_path(result_dir)

    profile_config = {
        "model_name": args.model_name,
        "result_dir": result_dir,
        "case_path": case_path,
        "dry_run": args.dry_run,
    }
    instance_config = {
        "stream": True,
        "temperature": args.temperature,
        "best_of": args.best_of,
        "top_k": args.top_k,
        "model_name": args.model_name,
        "openai_api_key": args.openai_api_key,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
    }
    scheduler_config = {
        "num_instances": args.num_instances,
        "instance_configurations": args.instance_configurations,
        "scheduler_policy": "prism",
        "scheduler_param": args.scheduler_param,
        "scaler_policy": "prism" if args.enable_prism_scaler else "none",
        "scaler_interval": args.scaler_interval,
        "req_predictor_policy": "prism_perceptron",
        "req_predictor_model_path": args.req_predictor_model_path,
    }
    prism_config = load_prism_config(args.prism_config)
    return PrismBenchmarkConfig(
        request_config=request_config,
        profile_config=profile_config,
        instance_config=instance_config,
        scheduler_config=scheduler_config,
        prism_config=prism_config,
    )


async def send_request_at_timestamp(request_id, request, sleeptime, dry_run):
    """Dispatch one request at its generated timestamp."""

    if not dry_run:
        await asyncio.sleep(float(sleeptime))
    global scheduler
    st = time.time()
    result = await scheduler.handle_request(request_id, request[:3])
    result["frontend_latency"] = time.time() - st
    return result


async def benchmark(requests, dry_run):
    """Run the asynchronous PRISM benchmark loop."""

    tasks = [
        asyncio.create_task(send_request_at_timestamp(request_id, request, request[3] + 1, dry_run))
        for request_id, request in enumerate(requests)
    ]
    results = []
    for completed_task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        results.append(await completed_task)
    return results


async def main():
    args = parse_args()
    configure_logging(args.log_level)
    config = build_config(args)
    logger.info(
        "Starting PRISM benchmark: requests=%d instances=%d dry_run=%s scaler=%s",
        config.request_config["request_num"],
        config.scheduler_config["num_instances"],
        args.dry_run,
        args.enable_prism_scaler,
    )

    if args.dry_run and config.request_config["load_mode"] == "random_generated":
        requests = generate_synthetic_requests(config.request_config)
        logger.info("Generated %d deterministic synthetic requests.", len(requests))
    else:
        from LLMServe.request_generater import Generator

        request_generator = Generator(
            request_config=config.request_config,
            model_name=config.instance_config["model_name"],
        )
        requests = request_generator.generate_requests()
        logger.info("Generated %d requests through the shared workload generator.", len(requests))

    global scheduler
    scheduler = PrismScheduler(
        scheduler_config=config.scheduler_config,
        instance_config=config.instance_config,
        prism_config=config.prism_config,
        dry_run=args.dry_run,
    )

    scaler = None
    if args.enable_prism_scaler and not args.dry_run:
        anticipator = PeakWorkloadAnticipator(
            peak_memory_capacity_tokens=config.prism_config.peak_memory_capacity_tokens,
            scale_up_memory_threshold=config.prism_config.scale_up_memory_threshold,
            scale_down_memory_threshold=config.prism_config.scale_down_memory_threshold,
            scaling_violation_tolerance=config.prism_config.scaling_violation_tolerance,
        )
        scaler = PrismScaler(
            scheduler=scheduler,
            anticipator=anticipator,
            interval=config.prism_config.adaptive_scaling_interval,
            cold_start_time=config.prism_config.instance_cold_start_seconds,
            scale_freeze_time=config.prism_config.scaling_cooldown_seconds,
        )
        await scaler.monitor_start()
        logger.info("PRISM adaptive scaler started.")

    results = await benchmark(requests, dry_run=args.dry_run)

    if scaler is not None:
        await scaler.monitor_stop()
        logger.info("PRISM adaptive scaler stopped.")

    save_prism_result(results, config)
    logger.info("Saved %d PRISM results to %s.", len(results), config.profile_config["case_path"])
    print("PRISM benchmark saved %d results to %s." % (len(results), config.profile_config["case_path"]))


def generate_synthetic_requests(request_config):
    """Generate deterministic synthetic requests for reproducible PRISM checks."""

    import random

    rng = random.Random(request_config["seed"])
    request_num = request_config["request_num"]
    prompt_mean = request_config["random_prompt_lens_mean"]
    prompt_range = request_config["random_prompt_lens_range"]
    response_mean = request_config["random_response_lens_mean"]
    response_range = request_config["random_response_lens_range"]

    def sample_len(mean, span):
        if span <= 0:
            return max(1, int(mean))
        low = max(1, int(mean - span // 2))
        high = max(low, int(mean + span // 2))
        return rng.randint(low, high)

    now = 0.0
    requests = []
    for _ in range(request_num):
        prompt_len = sample_len(prompt_mean, prompt_range)
        response_len = sample_len(response_mean, response_range)
        prompt = " ".join(["hello"] * max(1, prompt_len - 1))
        if request_config["workload"] == "uniform":
            now += 1.0 / max(1e-6, request_config["qps"])
        else:
            now += rng.expovariate(max(1e-6, request_config["qps"]))
        requests.append((prompt, prompt_len, response_len, now))
    return requests


def save_prism_result(results, config):
    """Persist PRISM benchmark results and the full experiment configuration."""

    result_dir = config.profile_config["result_dir"]
    os.makedirs(result_dir, exist_ok=True)
    case_path = config.profile_config["case_path"]
    os.makedirs(case_path, exist_ok=True)

    result_file = os.path.join(case_path, "result_%s.json" % time.strftime("%Y%m%d_%H%M%S"))
    with open(result_file, "w", encoding="utf-8") as f:
        for result in results:
            json.dump(result, f)
            f.write("\n")

    config_file = os.path.join(case_path, "config.json")
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump({
            "request_config": config.request_config,
            "profile_config": config.profile_config,
            "instance_config": config.instance_config,
            "scheduler_config": config.scheduler_config,
            "prism_config": config.prism_config.to_dict(),
        }, f, indent=4)


def next_case_path(result_dir):
    """Create the next ``case_N`` result directory."""

    os.makedirs(result_dir, exist_ok=True)
    case_numbers = []
    for name in os.listdir(result_dir):
        if name.startswith("case_"):
            try:
                case_numbers.append(int(name.split("_", 1)[1]))
            except ValueError:
                pass
    case_id = max(case_numbers) + 1 if case_numbers else 1
    path = os.path.join(result_dir, "case_%d" % case_id)
    while True:
        path = os.path.join(result_dir, "case_%d" % case_id)
        try:
            os.makedirs(path, exist_ok=False)
            return path
        except FileExistsError:
            case_id += 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the PRISM workload-aware LMaaS management prototype.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    prism_group = parser.add_argument_group("PRISM")
    prism_group.add_argument("--prism_config", type=str, default="prism_config.json", help="Path to PRISM configuration JSON.")
    prism_group.add_argument("--dry_run", action="store_true", help="Run prediction, routing, and result recording without contacting vLLM.")
    prism_group.add_argument("--enable_prism_scaler", action="store_true", help="Enable PRISM adaptive tuning during live serving.")
    prism_group.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Console logging level.")

    request_group = parser.add_argument_group("Request generation")
    request_group.add_argument("--request_num", type=int, default=100, help="Number of requests to generate.")
    request_group.add_argument("--benchmark_duration", type=int, default=None, help="Optional trace cutoff in seconds.")
    request_group.add_argument("--load", type=str, default="uniform", help="Prompt/response length source.")
    request_group.add_argument("--load_dataset_path", type=str, default=None, help="Path to prompt or load dataset.")
    request_group.add_argument("--seed", type=int, default=7, help="Random seed for reproducible request generation.")
    request_group.add_argument("--random_prompt_lens_mean", type=int, default=256, help="Mean prompt length for synthetic loads.")
    request_group.add_argument("--random_prompt_lens_range", type=int, default=64, help="Prompt length range for synthetic loads.")
    request_group.add_argument("--random_response_lens_mean", type=int, default=256, help="Mean response length for synthetic loads.")
    request_group.add_argument("--random_response_lens_range", type=int, default=64, help="Response length range for synthetic loads.")
    request_group.add_argument("--mixing_propotion", type=float, default=0.5, help="Compatibility option for mixed load generation.")

    workload_group = parser.add_argument_group("Arrival workload")
    workload_group.add_argument("--workload", type=str, default="uniform", help="Arrival process or trace name.")
    workload_group.add_argument("--workload_trace_path", type=str, default=None, help="Path to arrival trace.")
    workload_group.add_argument("--workload_timescale", type=float, default=1.0, help="Trace time compression factor.")
    workload_group.add_argument("--workload_sampling_interval", type=int, default=1, help="Trace sampling interval.")
    workload_group.add_argument("--workload_trace_range_L", type=float, default=0.0, help="Left bound of normalized trace range.")
    workload_group.add_argument("--workload_trace_range_R", type=float, default=1.0, help="Right bound of normalized trace range.")
    workload_group.add_argument("--qps", type=float, default=1.0, help="Queries per second for generated arrivals.")
    workload_group.add_argument("--coefficient_variation", type=float, default=0.0, help="Coefficient of variation for gamma arrivals.")

    serving_group = parser.add_argument_group("Serving")
    serving_group.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B", help="Served model name.")
    serving_group.add_argument("--result_dir", type=str, default="../results/", help="Directory for benchmark cases.")
    serving_group.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    serving_group.add_argument("--best_of", type=float, default=1, help="Number of completions for vLLM best_of.")
    serving_group.add_argument("--top_k", type=float, default=1, help="Top-k sampling setting.")
    serving_group.add_argument("--openai_api_key", type=str, default="token-abc123", help="OpenAI-compatible API key for vLLM.")
    serving_group.add_argument("--max_num_batched_tokens", type=float, default=8192, help="vLLM max_num_batched_tokens setting.")
    serving_group.add_argument("--max_model_len", type=float, default=4096, help="Maximum model context length.")
    serving_group.add_argument("--max_num_seqs", type=float, default=128, help="Maximum active sequences per instance.")

    system_group = parser.add_argument_group("Instance management")
    system_group.add_argument("--num_instances", type=int, default=1, help="Initial number of active instances.")
    system_group.add_argument("--instance_configurations", type=str, default="instance_configurations_4.json", help="Instance slot configuration file.")
    system_group.add_argument("--scheduler_param", type=float, default=4, help="Compatibility parameter retained for shared result configs.")
    system_group.add_argument("--scaler_interval", type=float, default=300, help="Compatibility interval retained for shared result configs.")
    system_group.add_argument("--req_predictor_model_path", type=str, default="../saved_model/1_claases-1_prompt-1_resample.pth", help="Compatibility path retained for shared result configs.")
    return parser.parse_args()


def configure_logging(log_level):
    """Configure concise console logging for PRISM commands."""

    logging.basicConfig(
        level=getattr(logging, log_level),
        format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
    )


if __name__ == "__main__":
    asyncio.run(main())
