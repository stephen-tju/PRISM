import os
import aiohttp
import re
import json
from datetime import datetime
import re
import pandas as pd
import matplotlib.pyplot as plt
# from LLMServe.logger import init_logger
# logger = init_logger()

case_folder = None


def read_dataset(path):
    df = pd.read_csv(path)
    return df


# Randomly mix from two different distributions
def merge_lists_by_ratio(L1, L2, ratio):
    count1 = round(ratio * 10)
    count2 = 10 - count1
    result = []
    i, j = 0, 0
    len1, len2 = len(L1), len(L2)
    
    while i < len1 or j < len2:
        for _ in range(count1):
            if i < len1:
                result.append(L1[i])
                i += 1
            else:
                break
        for _ in range(count2):
            if j < len2:
                result.append(L2[j])
                j += 1
            else:
                break

    return result


async def get_openai_metrics_text(host, port):
    metrics_url = f"http://{host}:{port}/metrics"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(metrics_url) as resp:
                if resp.status != 200:
                    print(f"Failed to fetch metrics: {resp.status} {resp.reason}")
                    return ""
                metrics_text = await resp.text()
                return metrics_text
        except Exception as e:
            print(f"Failed to fetch metrics text: {e}")
            return ""


# def parse_openai_metrics(metrics_text):
#     ''' 
#     exmaple:
#         # HELP vllm:gpu_cache_usage_perc GPU KV-cache usage. 1 means 100 percent usage.
#         # TYPE vllm:gpu_cache_usage_perc gauge
#         vllm:gpu_cache_usage_perc{model_name="meta-llama/Llama-2-7b-hf"} 0.0
#     pattern:
#         # ....
#         XX:XXX{...} 1.13123e+5

#     return:
#         {
#             "vllm:metric_name1": {
#                 "label1_in_str": 1.131,
#                 "label2_in_str": 1.131,
#             },
#             "vllm:metric_name2": {
#                 "label1_in_str": 1.131,
#             },
#         }
#     '''
#     # metric_pattern = re.compile(r'^# (.+)$|^(\w+:\w+){.+} (\d+\.\d+)$')
#     # metric_pattern = re.compile(r'^# (.+)$|^(\w+:\w+)\{([^}]+)\} (\d+\.\d+e[\+\-]\d+|\d+\.\d+|\d+)$')
#     metric_pattern = re.compile(r'^# (.+)$|^(\w+:\w+)\{([^}]+)\} (\d+\.\d+(e[+-]\d+)?)$')
#     ##                            comment | metric      labels    value

#     metrics_dict = {}
#     lines = metrics_text.split('\n')
#     for line in lines:
#         match = metric_pattern.match(line)
#         if match:
#             if match.group(1):
#                 continue

#             # ????
#             logger(match.group(2), match.group(3), match.group(4))
#             metric_name = match.group(2)
#             metric_label = match.group(3)
#             metric_value = float(match.group(4))

#             if metric_name not in metrics_dict:
#                 metrics_dict[metric_name] = {}
#             metrics_dict[metric_name][metric_label] = metric_value

#     return metrics_dict  


def parse_openai_metrics_single(metrics_text):
    """
    Faster version of parse_openai_metrics for single value metric
    """
    metric_pattern = re.compile(r'^# (.+)$|^(\w+:\w+)\{([^}]+)\} ([\+\-]\d+\.\d+(e[+-]\d+)?)$')
    metric_pattern = re.compile(r'^# (.+)$|^(\w+:\w+)\{([^}]+)\} ([+-]?\d+\.\d+(e[+-]\d+)?)$')
    ##                            comment | metric      labels    value
    
    metrics_dict = {}
    lines = metrics_text.split('\n')
    for line in lines[:]:
        match = metric_pattern.match(line)
        if match:
            if match.group(1):
                continue
            metric_name = match.group(2)
            metric_value = float(match.group(4))
            metrics_dict[metric_name] = metric_value  # overwrite if duplicate

    return metrics_dict


def parse_openai_metrics_TTFT_acc(metrics_text):
    # vllm:time_to_first_token_seconds_sum{model_name="meta-llama/Llama-2-7b-hf"} 1081.9915037155151
    pattern = re.compile(
        r'^vllm:time_to_first_token_seconds_sum\{model_name="[^"]+"\}\s+([0-9\.]+)$'
    )
    for line in metrics_text.splitlines():
        line = line.strip()
        match = pattern.match(line)
        if match:
            return float(match.group(1))
    return None

def parse_openai_metrics_req_acc(metrics_text):
    # vllm:request_success_total{finished_reason="length",model_name="meta-llama/Llama-2-7b-hf"} 1998.0
    pattern = re.compile(
        r'^vllm:request_success_total\{finished_reason="length",model_name="[^"]+"\}\s+([0-9\.]+)$'
    )
    for line in metrics_text.splitlines():
        line = line.strip()
        match = pattern.match(line)
        if match:
            return float(match.group(1))
    return None



def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

def create_next_case_folder(base_path):
    case_pattern = re.compile(r"^case_(\d+)$")
    subfolders = [
        f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))
    ]

    case_numbers = []
    for folder in subfolders:
        match = case_pattern.match(folder)
        if match:
            case_numbers.append(int(match.group(1)))

    if len(case_numbers):
        max_case_number = max(case_numbers)
    else:
        max_case_number = 0

    new_case_number = max_case_number + 1
    new_case_folder = f"case_{new_case_number}"

    new_case_path = os.path.join(base_path, new_case_folder)
    os.makedirs(new_case_path)

    # logger.info(f"Saving new case {new_case_number} to folder {new_case_path}")
    return new_case_path


# Expecting 'result_dir' to be an existing directory
def get_case_path(result_dir):
    global case_folder
    if case_folder is None:
        case_folder = create_next_case_folder(result_dir)
    
    return case_folder


def save_benchmark_result(results, config):
    result_dir = config.profile_config["result_dir"]
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    new_case_path = get_case_path(result_dir)
    config.profile_config["case_path"] = new_case_path

    # Save results to file
    result_file = os.path.join(
        new_case_path, f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(result_file, "w") as f:
        for result in results:
            json.dump(result, f)
            f.write("\n")

    # Save config to file
    config_file = os.path.join(
        new_case_path, "config.json"
    )
    config_dict = {
        "request_config": config.request_config,
        "profile_config": config.profile_config,
        "instance_config": config.instance_config,
        "scheduler_config": config.scheduler_config,
    }
    with open(config_file, "w") as f:
        json.dump(config_dict, f, indent=4)
    
    # return new_case_path


def load_instance_configuration(instance_config_file):
    # Abort if failed
    with open(instance_config_file, "r") as f:
        instance_config = json.load(f)

    return instance_config


def get_tok_id_lens(tokenizer, text):
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    return len(token_ids)


def analyze_vllm_log_without_preemption(log_file, result_dir):
    log_pattern = re.compile(
        r"INFO (\d{2}-\d{2} \d{2}:\d{2}:\d{2}) metrics\.py:\d+\] "
        r"Avg prompt throughput: ([\d.]+) tokens/s, "
        r"Avg generation throughput: ([\d.]+) tokens/s, "
        r"Running: (\d+) reqs, Swapped: (\d+) reqs, Pending: (\d+) reqs, "
        r"GPU KV cache usage: ([\d.]+)%, CPU KV cache usage: ([\d.]+)%."
    )

    with open(log_file, 'r') as file:
        log_lines = file.readlines()
    
    data = []
    for line in log_lines:
        if 'Avg prompt throughput' not in line:
            continue
        match = log_pattern.match(line)
        if match:
            timestamp_str = match.group(1)
            timestamp = datetime.strptime(timestamp_str, '%m-%d %H:%M:%S')
            avg_prompt_throughput = float(match.group(2))
            avg_generation_throughput = float(match.group(3))
            running_reqs = int(match.group(4))
            swapped_reqs = int(match.group(5))
            pending_reqs = int(match.group(6))
            gpu_kv_cache_usage = float(match.group(7))
            cpu_kv_cache_usage = float(match.group(8))
            
            data.append({
                'timestamp': timestamp,
                'avg_prompt_throughput': avg_prompt_throughput,
                'avg_generation_throughput': avg_generation_throughput,
                'running_reqs': running_reqs,
                'swapped_reqs': swapped_reqs,
                'pending_reqs': pending_reqs,
                'gpu_kv_cache_usage': gpu_kv_cache_usage,
                'cpu_kv_cache_usage': cpu_kv_cache_usage
            })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(df.index, df['avg_prompt_throughput'], label='Avg Prompt Throughput (tokens/s)')
    axes[0].plot(df.index, df['avg_generation_throughput'], label='Avg Generation Throughput (tokens/s)')
    axes[0].set_ylabel('Throughput (tokens/s)')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(df.index, df['running_reqs'], label='Running Requests')
    axes[1].plot(df.index, df['swapped_reqs'], label='Swapped Requests')
    axes[1].plot(df.index, df['pending_reqs'], label='Pending Requests')
    axes[1].set_ylabel('Requests')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(df.index, df['gpu_kv_cache_usage'], label='GPU KV Cache Usage (%)')
    axes[2].plot(df.index, df['cpu_kv_cache_usage'], label='CPU KV Cache Usage (%)')
    axes[2].set_ylabel('KV Cache Usage (%)')
    axes[2].set_xlabel('Time')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    figure_path = os.path.join(result_dir, 'metrics.png')
    plt.savefig(figure_path)


def analyze_vllm_log(log_file, result_dir):
    log_pattern = re.compile(
        r"INFO (\d{2}-\d{2} \d{2}:\d{2}:\d{2}) metrics\.py:\d+\] "
        r"Avg prompt throughput: ([\d.]+) tokens/s, "
        r"Avg generation throughput: ([\d.]+) tokens/s, "
        r"Running: (\d+) reqs, Swapped: (\d+) reqs, Pending: (\d+) reqs, "
        r"GPU KV cache usage: ([\d.]+)%, CPU KV cache usage: ([\d.]+)%."
    )
    preemption_pattern = re.compile(r"(\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*is preempted")
    added_request_pattern = re.compile(r"INFO (\d{2}-\d{2} \d{2}:\d{2}:\d{2}) async_llm_engine.py:\d+\] Added request")
    finished_request_pattern = re.compile(r"INFO (\d{2}-\d{2} \d{2}:\d{2}:\d{2}) async_llm_engine.py:\d+\] Finished request")
    
    data = []
    preemption_data = []
    request_data = []
    
    total_requests = 0
    total_finished_requests = 0

    with open(log_file, 'r') as file:
        log_lines = file.readlines()

    for line in log_lines:
        # Track preemptions
        if "is preempted" in line:
            match = preemption_pattern.search(line)
            if match:
                timestamp_str = match.group(1)
                timestamp = datetime.strptime(timestamp_str, '%m-%d %H:%M:%S')
                preemption_data.append({'timestamp': timestamp, 'preemption_cnt': 1})
        
        # Track added requests
        elif "Added request" in line:
            match = added_request_pattern.search(line)
            if match:
                timestamp_str = match.group(1)
                timestamp = datetime.strptime(timestamp_str, '%m-%d %H:%M:%S')
                total_requests += 1
                request_data.append({
                    'timestamp': timestamp,
                    'total_requests': total_requests,
                    'total_finished_requests': total_finished_requests
                })
        
        # Track finished requests
        elif "Finished request" in line:
            match = finished_request_pattern.search(line)
            if match:
                timestamp_str = match.group(1)
                timestamp = datetime.strptime(timestamp_str, '%m-%d %H:%M:%S')
                total_finished_requests += 1
                request_data.append({
                    'timestamp': timestamp,
                    'total_requests': total_requests,
                    'total_finished_requests': total_finished_requests
                })

        elif 'Avg prompt throughput' not in line:
            continue
        
        # Match other log patterns
        match = log_pattern.match(line)
        if match:
            timestamp_str = match.group(1)
            timestamp = datetime.strptime(timestamp_str, '%m-%d %H:%M:%S')
            avg_prompt_throughput = float(match.group(2))
            avg_generation_throughput = float(match.group(3))
            running_reqs = int(match.group(4))
            swapped_reqs = int(match.group(5))
            pending_reqs = int(match.group(6))
            gpu_kv_cache_usage = float(match.group(7))
            cpu_kv_cache_usage = float(match.group(8))
            
            data.append({
                'timestamp': timestamp,
                'avg_prompt_throughput': avg_prompt_throughput,
                'avg_generation_throughput': avg_generation_throughput,
                'running_reqs': running_reqs,
                'swapped_reqs': swapped_reqs,
                'pending_reqs': pending_reqs,
                'gpu_kv_cache_usage': gpu_kv_cache_usage,
                'cpu_kv_cache_usage': cpu_kv_cache_usage
            })

    # Convert to DataFrames
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    preemption_df = pd.DataFrame(preemption_data)
    if not preemption_df.empty:
        preemption_df.set_index('timestamp', inplace=True)
        preemption_df = preemption_df.resample('T').sum().fillna(0)  # Summing by minute intervals, fill missing with 0

    request_df = pd.DataFrame(request_data)
    request_df.set_index('timestamp', inplace=True)
    request_df = request_df[~request_df.index.duplicated(keep='last')]
    request_df = request_df.resample('T').ffill().fillna(0)  # Forward fill to keep the cumulative count up-to-date

    # Plotting
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

    axes[0].plot(df.index, df['avg_prompt_throughput'], label='Avg Prompt Throughput (tokens/s)')
    axes[0].plot(df.index, df['avg_generation_throughput'], label='Avg Generation Throughput (tokens/s)')
    axes[0].set_ylabel('Throughput (tokens/s)')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(df.index, df['running_reqs'], label='Running Requests')
    axes[1].plot(df.index, df['swapped_reqs'], label='Swapped Requests')
    axes[1].plot(df.index, df['pending_reqs'], label='Pending Requests')
    axes[1].set_ylabel('Requests')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(df.index, df['gpu_kv_cache_usage'], label='GPU KV Cache Usage (%)')
    axes[2].plot(df.index, df['cpu_kv_cache_usage'], label='CPU KV Cache Usage (%)')
    axes[2].set_ylabel('KV Cache Usage (%)')
    axes[2].legend()
    axes[2].grid(True)

    # Plotting request counts and optionally preemption counts with dual y-axes
    ax4 = axes[3]
    ax4.plot(request_df.index, request_df['total_requests'], label='Total Requests', color='blue')
    ax4.plot(request_df.index, request_df['total_finished_requests'], label='Total Finished Requests', color='green')
    ax4.set_ylabel('Requests Count')
    ax4.set_xlabel('Time')
    ax4.legend(loc="upper left")
    ax4.grid(True)

    # Add twin y-axis for preemption count if data is available
    if not preemption_df.empty:
        ax4_twin = ax4.twinx()
        ax4_twin.plot(preemption_df.index, preemption_df['preemption_cnt'], label='Preemption Count', color='purple', linestyle='--')
        ax4_twin.set_ylabel('Preemption Count')
        ax4_twin.legend(loc="upper right")

    plt.tight_layout()
    figure_path = os.path.join(result_dir, 'metrics_with_requests_preemption.png')
    plt.savefig(figure_path)
