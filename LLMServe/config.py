import argparse
import os
import time
from LLMServe.util import get_case_path
from LLMServe.logger import init_logger

# logger = init_logger()
_config = None

load_choices = {
    "prompt_dataset": [
        "ShareGPT",
        "LMSYS-Chat-1M",
    ],
    "random_generated": [
        "uniform",
        "uniform_mixing",
        "exponential",
        "capped_exponential",
        "zipf",
    ],
    "workload_trace_sample": [
        "BurstGPT",
        "Azure_code",
        "Azure_conv",
    ],
    "workload_trace": [
        "workload_trace",
    ],
}
workload_choices = {
    "random_generated": [
        "uniform",
        "gamma",
        "poisson",
    ],
    "trace": [
        "BurstGPT",
        "Azure_code",
        "Azure_code_peak",
        "Azure_conv",
        "Azure_conv_peak",
    ],
}
# scaler_choices = [
#     "none",
#     # "TTFT_threshold",
#     # "memory_threshold",
#     "reactive",
#     # "workload_predictor",
#     # "workload_monitor",
#     "proactive",
#     "proactive_dry",
#     "hybrid",
#     # "scheduler_load",
#     "scheduler_lookahead",
#     "preserve",
# ]


def add_request_config(parser):
    parser.add_argument(
        "--request_num",
        type=int,
        default=100,
        help="Number of requests.",
    )
    parser.add_argument(
        "--benchmark_duration",
        type=int,
        default=None,
        help="Duration of the benchmark in seconds (Now only for workload_trace).",    
    )

    # Load Config
    parser.add_argument(
        "--load",
        type=str,
        default="ShareGPT",
        # all in load_choices
        choices=load_choices["prompt_dataset"] + 
                load_choices["random_generated"] + 
                load_choices["workload_trace_sample"] +
                load_choices["workload_trace"],
        help="The name of the input load dataset.",
    )
    parser.add_argument(
        "--load_dataset_path",
        type=str,
        default=None,
        help="The path to the prompt dataset / workload trace dataset.",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=int(time.time()), 
        help="Random seed for prompt set shuffling."
    )
    # If using random generated data:
    parser.add_argument(
        "--random_prompt_lens_mean",
        type=int,
        default=0,
        help="Mean length of random prompts.",
    )
    parser.add_argument(
        "--random_prompt_lens_range",
        type=int,
        default=0,
        help="Range of random prompt lengths.",
    )
    parser.add_argument(
        "--random_response_lens_mean",
        type=int,
        default=0,
        help="Mean length of random responses.",
    )
    parser.add_argument(
        "--random_response_lens_range",
        type=int,
        default=0,
        help="Range of random response lengths.",
    )
    parser.add_argument(
        "--mixing_propotion",
        type=float,
        default=0.5,
        help="The propotion of the first part of the mixing prompts.",
    )

    # Workload Config
    parser.add_argument(
        "--workload",
        type=str,
        default="uniform",
        choices=workload_choices["random_generated"] + 
                workload_choices["trace"],
        help="Type of workload. Options: Azure, uniform, gamma, poisson",
    )
    parser.add_argument(
        "--workload_trace_path",
        type=str,
        default=None,
        help="The path to the workloads.",
    )
    parser.add_argument(
        "--workload_timescale",
        type=float,
        default=1.0,
        help="Scale trace timely. 100 means 100 times faster.",
    )
    parser.add_argument(
        "--workload_sampling_interval",
        type=int,
        default=1.0,
        help="Sampling interval for workload trace. 100 means sample 1 from every 100 reqs.", 
    )
    parser.add_argument(
        "--workload_trace_range_L",
        type=float,
        default=0.0,
        help="The left bound of the trace range.",
    )
    parser.add_argument(
        "--workload_trace_range_R",
        type=float,
        default=1.0,
        help="The right bound of the trace range.",
    )
    parser.add_argument(
        "--qps", 
        type=float, 
        default=1.0,
        help="Queries per second."
    )
    # If gamma
    parser.add_argument(
        "--coefficient_variation",
        type=float,
        default=0.0,
        help="Coefficient of variation for gamma distribution.",
    )

def add_profile_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
        help="The name of the served model.",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="../results/",
        help="Path to save the request serving result file.",
    )



def add_instance_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--stream",
        action="store_true",
        default=True,
        help="Enable streaming output in server mode.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for model generation.",
    )
    parser.add_argument(
        "--best_of",
        type=float,
        default=1,
        help="Number of outputs to generate and return the best of.",
    )
    parser.add_argument(
        "--top_k",
        type=float,
        default=1,
        help="Number of highest probability vocabulary tokens to keep for top-k sampling.",
    )
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default="token-abc123",
        help="OpenAI API key for OpenAI backend.",
    )
    # parser.add_argument(
    #     "--vllm_log",
    #     type=str,
    #     default="/root/vllm.log",
    #     help="Path to the vLLM log file.",
    # )
    parser.add_argument(
        "--max_num_batched_tokens",
        type=float,
        default=8192,
    )
    parser.add_argument(
        "--max_model_len",
        type=float,
        default=4096,
    )
    parser.add_argument(
        "--max_num_seqs",
        type=float,
        default=128,
    )


def add_scheduler_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--num_instances",
        type=int,
        default=1,
        help="Initial number of instances used to serve requests.",
    )
    parser.add_argument(
        "--instance_configurations",
        type=str,
        default="../instance_configurations.json",
        help="Path to the configurations file of all available instance.",
    )
    parser.add_argument(
        "--scheduler_policy",
        type=str,
        # choices=["least_loaded", "round_robin", "random", "with_ground_truth"],
        default="round_robin",
        help="The scheduling policy for instance selection.",
    )
    parser.add_argument(
        "--scheduler_param",
        type=float,
        default=4,
        help="The parameter for the scheduler policy (load metric caculation).",
    )
    parser.add_argument(
        "--req_predictor_policy",
        type=str,
        choices=["ground_truth", "load_predictor"],
        default="ground_truth",
        help="The request predictor policy for instance scaling.",
    )
    parser.add_argument(
        "--req_predictor_model_path",
        type=str,
        default="../saved_model/1_claases-1_prompt-1_resample.pth",
        help="The saved model path for the load predictor",
    )
    parser.add_argument(
        "--scaler_policy",
        type=str,
        # choices=scaler_choices,
        default="none",
        help="The scaling policy for instance scaling.",
    )
    parser.add_argument(
        "--scaler_interval",
        type=float,
        default=300,
        help="The interval (seconds) for the scaler monitor.",
    )


class Config:
    def __init__(self, request_config, profile_config, instance_config, scheduler_config):
        self.request_config = request_config
        self.profile_config = profile_config
        self.instance_config = instance_config
        self.scheduler_config = scheduler_config


def get_all_config() -> Config:
    global _config
    if _config is not None:
        return _config

    parser = argparse.ArgumentParser()
    add_request_config(parser)
    add_profile_config(parser)
    add_instance_config(parser)
    add_scheduler_config(parser)
    args = parser.parse_args()

    load_mode = [k for k, v in load_choices.items() if args.load in v][0]
    workload_mode = [k for k, v in workload_choices.items() if args.workload in v][0]
    
    request_config = {
        "request_num": args.request_num,
        "benchmark_duration": args.benchmark_duration,
        # load
        "load": args.load,
        "load_dataset_path": args.load_dataset_path,
        "seed": args.seed,
        "random_prompt_lens_mean": args.random_prompt_lens_mean,
        "random_prompt_lens_range": args.random_prompt_lens_range,
        "random_response_lens_mean": args.random_response_lens_mean,
        "random_response_lens_range": args.random_response_lens_range,
        "mixing_propotion": args.mixing_propotion,
        # workload
        "workload": args.workload,
        "workload_trace_path": args.workload_trace_path,
        "workload_timescale": args.workload_timescale,
        "workload_sampling_interval": args.workload_sampling_interval,
        "workload_trace_range_L": args.workload_trace_range_L,
        "workload_trace_range_R": args.workload_trace_range_R,
        "qps": args.qps,
        "coefficient_variation": args.coefficient_variation,
        # Overall
        "load_mode": load_mode,
        "workload_mode": workload_mode,
    }

    # Load
    if load_mode == "prompt_dataset":
        if args.load_dataset_path == None:
            request_config["load_dataset_path"] = os.path.join(
                "../data/datasets/", os.path.join(args.load, "cleaned.csv")
            )
    else: 
        if args.req_predictor_policy == "load_predictor":
            raise ValueError("Load predictor is not applicable without prompt dataset.")
            
    # Workload
    if workload_mode == "trace":
        if args.workload_trace_path is None:
            request_config["workload_trace_path"] = os.path.join(
                "../data/workloads/", os.path.join(args.workload, args.workload + ".csv")
            )
    else:  # random generated workload
        if load_mode == "workload_trace":
            raise ValueError("Workload trace is not used, but the load is set to workload_trace.") 

    profile_config = {
        "model_name": args.model_name,
        "result_dir": args.result_dir,
        "case_path": None,
    }

    result_dir = profile_config["result_dir"]
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    case_path = get_case_path(result_dir)
    profile_config["case_path"] = case_path

    # The actual initialization of logger
    global logger
    logger = init_logger(
        log_file_path=os.path.join(case_path, "run.log")
    )

    instance_config = {
        "stream": args.stream,
        "temperature": args.temperature,
        "best_of": args.best_of,
        "top_k": args.top_k,
        "model_name": args.model_name,
        "openai_api_key": args.openai_api_key,
        # "vllm_log": args.vllm_log,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
    }

    scheduler_config = {
        "num_instances": args.num_instances,
        "instance_configurations": args.instance_configurations,
        "scheduler_policy": args.scheduler_policy,
        "scheduler_param": args.scheduler_param,
        "scaler_policy": args.scaler_policy,
        "scaler_interval": args.scaler_interval,
        "req_predictor_policy": args.req_predictor_policy,
        "req_predictor_model_path": args.req_predictor_model_path,
    }

    # Scaler
    if scheduler_config["scaler_policy"] == "workload_predictor":
        if workload_mode != "trace":
            raise ValueError("Workload predictor is only applicable with workload trace.")
        if load_mode != "workload_trace":
            raise ValueError("Workload predictor is only applicable with workload trace.")

    _config = Config(request_config, profile_config, instance_config, scheduler_config)
    return _config


_config = get_all_config()
