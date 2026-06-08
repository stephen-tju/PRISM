import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import re
import pytz
import datetime
from itertools import chain
import argparse

MAX_NUM_INSTANCES = 8

def process_monitor_log_all(case_dir, all_metric_keys):
    log_path = os.path.join(case_dir, "monitor.log")
    try:
        with open(log_path, "r") as file:
            log_lines = file.readlines()
    except:
        print(f"No monitor log file: {log_path}")
        return None

    all_data = [ {key: {"time": [], "value": []} for key in all_metric_keys} for _ in range(MAX_NUM_INSTANCES) ]
    num_instances_data = {"time": [], "value": []}

    for line in log_lines:
        if "[INFO]" in line or "[ERROR]" in line:
            continue
        timestamp = re.findall(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}', line)[0]
        dt = datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S,%f')
        # timezone = pytz.FixedOffset(-7 * 60)  
        timezone = pytz.FixedOffset(0 * 60)  
        time_val = timezone.localize(dt).timestamp()
        try:
            log_type, log_contents = eval(line[35:])
        except:
            print(f"Error: unable to parse line: {line}")
            continue

        current_num_instances = len(log_contents)
        num_instances_data["time"].append(time_val)
        num_instances_data["value"].append(current_num_instances)

        for instance_id, log_content in enumerate(log_contents):
            for key in all_metric_keys:
                if key in log_content:
                    all_data[instance_id][key]["time"].append(time_val)
                    all_data[instance_id][key]["value"].append(log_content[key])

    for instance_id in range(MAX_NUM_INSTANCES):
        if "vllm:num_requests_running" in all_data[instance_id] and len(all_data[instance_id]["vllm:num_requests_running"]["value"]) > 0:
            base_value = all_data[instance_id]["vllm:num_preemptions_total"]["value"][0]
            all_data[instance_id]["vllm:num_preemptions_total"]["value"] = [
                value - base_value for value in all_data[instance_id]["vllm:num_preemptions_total"]["value"]
            ]
    return all_data, num_instances_data

def procecss_result_file(case_dir, result_file):
    result_path = os.path.join(case_dir, result_file)
    results = []
    try:
        with open(result_path, "r") as file:
            lines = file.readlines()
    except:
        print(f"Error: unable to read result file: {result_path}")
        return None

    for line in lines:
        record = json.loads(line.strip())
        results.append(record)
    return pd.DataFrame(results)

def aggregate_by_interval(timestamps, datas, interval, method='mean'):
    bins = np.floor(timestamps / interval).astype(int)
    df = pd.DataFrame({'timestamp': bins, 'datas': datas})
    aggregated_data = df.groupby('timestamp').agg({'datas': method})
    aggregated_timestamps = aggregated_data.index * interval  
    aggregated_data = aggregated_data.values.flatten()
    return aggregated_timestamps, aggregated_data

def read_monitor_data(case_dir):
    files = os.listdir(case_dir)
    result_files = [f for f in files if f.startswith("result") and f.endswith(".json")]
    if len(result_files) != 1:
        print(f"Error: found {len(result_files)} result file in case {case_dir}")
        return None
    result_file = result_files[-1]
    result_timestamp = result_file.split("_")[1].split(".")[0]
    
    df = procecss_result_file(case_dir, result_file)
    if df is None or 'TTFT' not in df.columns:
        print(f"Error: unable to process result file in case {case_dir}")
        return None

    with open(os.path.join(case_dir, "config.json"), "r") as file:
        config = json.load(file)
    request_config = config["request_config"]

    timestamp_base = df["record_time"].values[0]
    df["record_time"] = df["record_time"] - timestamp_base

    df_latency = df.dropna(subset=['TTFT', 'latency', 'record_time', 'TBPT']).copy()
    
    df_latency["ttft_timestamp"] = df_latency["itl"].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
    df_latency["ttft_timestamp"] = df_latency["ttft_timestamp"] - timestamp_base
    df_latency["TTFT_normalized"] = df_latency["TTFT"] / df_latency["prompt_tokens"]
    df_latency["latency_normalized"] = df_latency["latency"] / df_latency["generated_tokens"]

    subploting_map = {
        "Load": {
            "request_num": 0,
            "incoming_prefill_tokens": 1,
            "incoming_decode_tokens": 1,
            "expected_token_usage": 1,
            "current_token_usage": 1,
        },
        "Internal": {
            "vllm:num_requests_running": 0,
            "vllm:num_requests_waiting": 0,
            "vllm:num_preemptions_total": 0,
            "vllm:gpu_cache_usage_perc": 1,
        },
    }
    all_metric_keys = list(chain.from_iterable(subploting_map.values()))
    all_metric_subploting_ids = [value for sublist in subploting_map.values() for value in sublist.values()]

    all_monitor_data, instance_nums_data = process_monitor_log_all(case_dir, all_metric_keys)
    GPU_max_tokens = 16 * 2306

    return {
        "df_latency": df_latency,
        "timestamp_base": timestamp_base,
        "result_timestamp": result_timestamp,
        "all_metric_keys": all_metric_keys,
        "all_metric_subploting_ids": all_metric_subploting_ids,
        "all_monitor_data": all_monitor_data,
        "instance_nums_data": instance_nums_data,
        "GPU_max_tokens": GPU_max_tokens,
        "request_config": request_config,
    }
    

def get_data(case_dir):
    # case_dir = f"./case_{case_id}"
    if not os.path.exists(case_dir):
        return None, None
    data = read_monitor_data(case_dir)
    if data is None:
        return None, None
    with open(os.path.join(case_dir, "config.json"), "r") as file:
        config = json.load(file)
    df_latency = data["df_latency"]
    timestamp_base = data["timestamp_base"]
    # all_monitor_data = data["all_monitor_data"]
    instance_nums_data = data["instance_nums_data"]
    
    ttft_interval = 5.0
    ttft_agg_timestamp, ttft_agg_data = aggregate_by_interval(
        df_latency["ttft_timestamp"].dropna(), df_latency["TTFT"].dropna(), ttft_interval, method='mean'
    )
    
    norm_latency_interval = 5.0
    norm_latency_agg_timestamp, norm_latency_agg_data = aggregate_by_interval(
        df_latency["record_time"], df_latency["latency_normalized"], norm_latency_interval, method='mean'
    )

    # norm_latency_interval = 5.0
    # norm_latency_agg_timestamp, norm_latency_agg_data = aggregate_by_interval(
    #     df_latency["record_time"], df_latency["latency"], norm_latency_interval, method='mean'
        # )

    tps_interval = 10.0
    tps_timestamp, tps_data = aggregate_by_interval(
        df_latency["record_time"], df_latency["prompt_tokens"] + df_latency["generated_tokens"],
        tps_interval, method='sum'
    )
    instance_timestamp = instance_nums_data["time"] - timestamp_base,
    instance_timestamp = [t - timestamp_base for t in instance_nums_data["time"]]
    # instance_timestamp = instance_nums_data["time"] - timestamp_base,
    instance_data = instance_nums_data["value"]
    
    case_data = {
        "ttft_agg_timestamp": ttft_agg_timestamp,
        "ttft_agg_data": ttft_agg_data,
        "norm_latency_agg_timestamp": norm_latency_agg_timestamp,
        "norm_latency_agg_data": norm_latency_agg_data,
        "tps_agg_timestamp": tps_timestamp,
        "tps_agg_data": tps_data,
        "instance_timestamp": instance_timestamp,
        "instance_data": instance_data,
    }
    return case_data, config


def get_time_xticks(ax, value):
    step_per_hour = (60 * 60) // 24
    max_timestamp = max(value)
    tick_positions= list(np.arange(0, max_timestamp, 6 * step_per_hour))
    ax.set_xticks(tick_positions)
    tick_labels = []
    first_item = True
    for i in tick_positions:
        si = int(i / step_per_hour) % 24
        if si == 0:
            if first_item:
                first_item = False
            else:
                si = 24
        tick_labels.append(f"{si}")
    ax.set_xticklabels(tick_labels, fontsize=22)


def reverse_mapping(row_id: int, col_id: int):
    """
    reverse mapping from row_id and col_id to dataset_name and method
    
    Args:
        row_id: row index
        col_id: column index
    
    Returns:
        (dataset_name, method)
    
    Raises:
        ValueError: when the input row_id and col_id combination is invalid
    """
    # define the reverse mapping from row_id and col_id to dataset_name and method
    reverse_mappings = {
        (0, 1): ("Azure_code", "Reactive Auto-scaling"),
        (0, 2): ("Azure_code", "Proactive Auto-scaling"),
        (0, 3): ("Azure_code", "Hybrid (Proactive + Reactive)"),
        (1, 0): ("Azure_code", "PASS"),
        (1, 1): ("Azure_code", "Llumnix"),
        (1, 2): ("Azure_code", "PreServe (w/o Load Prediction Error)"),
        (1, 3): ("Azure_code", "PreServe (w/ Load Prediction Error)"),
        (2, 1): ("Azure_conv", "Reactive Auto-scaling"),
        (2, 2): ("Azure_conv", "Proactive Auto-scaling"),
        (2, 3): ("Azure_conv", "Hybrid (Proactive + Reactive)"),
        (3, 0): ("Azure_conv", "PASS"),
        (3, 1): ("Azure_conv", "Llumnix"),
        (3, 2): ("Azure_conv", "PreServe (w/o Load Prediction Error)"),
        (3, 3): ("Azure_conv", "PreServe (w/ Load Prediction Error)"),
    }
    
    # check if the input row_id and col_id combination is valid
    key = (row_id, col_id)
    if key not in reverse_mappings:
        raise ValueError(f"Invalid row_id and col_id combination: {row_id}, {col_id}")
    
    # return the reverse mapping result
    return reverse_mappings[key]

def plot_data(axs, row_id, col_id, case_data, config):
    # print(f"row_id: {row_id}, col_id: {col_id}, config: {config}")
    tps_colors = ["#b0d992", "#b0d992", "#99b9e9", "#99b9e9"]
    if (row_id == 0 and col_id == 1) or (row_id == 2 and col_id == 1):
        value = case_data["tps_agg_timestamp"]
        if row_id == 0:
            axs[row_id, 0].plot(value, case_data["tps_agg_data"], label="Azure-code TPS", color=tps_colors[row_id])
            axs[row_id, 0].set_yticks([100000, 200000, 300000])
            axs[row_id, 0].set_ylabel("Token-Per-Second", fontsize=28)
        else:
            axs[row_id, 0].plot(value, case_data["tps_agg_data"], label="Azure-chat TPS", color=tps_colors[row_id])
            axs[row_id, 0].set_yticks([50000, 100000, 150000])
            axs[row_id, 0].set_ylabel("Token-Per-Second", fontsize=28)
        # if row_id == 0:
        axs[row_id, 0].set_title("Workload & Timeline", fontsize=33)
        
        axs[row_id, 0].ticklabel_format(axis='y', style='sci', scilimits=(0,0)) 
        # if row_id == 1:
            # axs[row_id, 0].set_xlabel("Time (Hour)", fontsize=33)
        get_time_xticks(axs[row_id, 0], value)

    # if row_id % 2 == 0:
        # ax_primary = axs[row_id, col_id+1]
    # else:
    ax_primary = axs[row_id, col_id]
    # ax_primary.plot(case_data["norm_latency_agg_timestamp"], case_data["norm_latency_agg_data"], label="Norm. Latency", color="#f6b3b7")
    if row_id == 0 and col_id == 1:
        ax_primary.plot(case_data["norm_latency_agg_timestamp"], case_data["norm_latency_agg_data"], label="Normalized Latency", color="#ef6c5f")
    else:
        ax_primary.plot(case_data["norm_latency_agg_timestamp"], case_data["norm_latency_agg_data"], color="#ef6c5f")
    _, method_name = reverse_mapping(row_id, col_id)
    print(f"Method Name: {method_name}, Workload: {config['request_config']['workload']}, Max Latency: {max(case_data['norm_latency_agg_data'])}")
    
    # if row_id == 0:
    ax_primary.set_title(f"{method_name}", fontsize=33)
    value = case_data["norm_latency_agg_timestamp"]
    # ax_primary.set_ylabel("Norm. Latency")
    get_time_xticks(ax_primary, value)
    if row_id < 2:
        ax_primary.set_ylim(0, 1.05)
        ax_primary.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax_primary.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=26)
    else:
        ax_primary.set_ylim(0, 1.575)
        ax_primary.set_yticks([0, 0.3, 0.6, 0.9, 1.2, 1.5])
        ax_primary.set_yticklabels([0, 0.3, 0.6, 0.9, 1.2, 1.5], fontsize=26)
    
    ax_secondary = ax_primary.twinx()
    max_value = max(case_data["tps_agg_data"])
    normalized_tps = [8 * v / max_value for v in case_data["tps_agg_data"]]
    ax_secondary.plot(case_data["tps_agg_timestamp"], normalized_tps, color=tps_colors[row_id], alpha=0.4)
    avg_num_inst = sum(case_data["instance_data"]) / len(case_data["instance_data"])
    print("avg #inst", avg_num_inst, " reduced by", (8 - avg_num_inst) / 8.0)
    ax_secondary.set_ylim(-0.1, 8.25)
    ax_secondary.set_yticks([2, 4, 6, 8])
    ax_secondary.set_yticklabels([2, 4, 6, 8], fontsize=26)

    if row_id == 3:
        ax_primary.set_xlabel("Time (Hour)", fontsize=28)
    if col_id == 0:
        ax_primary.set_ylabel("Normalized Latency", fontsize=28)
    if col_id == 3:
        ax_secondary.set_ylabel("Instances", fontsize=28)

    if row_id == 0 and col_id == 1:
        ax_secondary.plot(case_data["instance_timestamp"], case_data["instance_data"], linestyle="--",label="Instance Number", color="#F09202", alpha=0.6)
        return ax_secondary.get_legend_handles_labels()
    else:
        ax_secondary.plot(case_data["instance_timestamp"], case_data["instance_data"], linestyle="--", color="#F09202", alpha=0.6)
        return None
    

    # ax_secondary.set_ylabel("Instances")
    
    
def get_mapping(dataset_name, method):
    """
    return the mapping from dataset_name and method to row_id and col_id
    
    Args:
        dataset_name: dataset name, e.g. "Azure_code" or "Azure_conv"
        method: method name, e.g. "reactive" or "proactive"
    
    Returns:
        (row_id, col_id)
    
    Raises:
        ValueError: when the input dataset_name or method is invalid
    """
    # define the mapping from dataset_name and method to row_id and col_id
    mappings = {
        "Azure_code": {
            "reactive": (0, 1), 
            "proactive": (0, 2),
            "hybrid": (0, 3),
            "pass": (1, 0),
            "llumnix": (1, 1),
            "preserve": (1, 2),
            "preserve_err": (1, 3),
        },
        "Azure_conv": {
            "reactive": (2, 1),
            "proactive": (2, 2),
            "hybrid": (2, 3),
            "pass": (3, 0),
            "llumnix": (3, 1),
            "preserve": (3, 2),
            "preserve_err": (3, 3),
        }
    }
    
    # check if the input dataset_name is valid
    if dataset_name not in mappings:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
    
    # check if the input method is valid
    if method not in mappings[dataset_name]:
        raise ValueError(f"Invalid method: {method}")
    
    # return the mapping result
    return mappings[dataset_name][method]


def plot_figures():
    plt.rcParams.update({'font.size': 26, "font.family": 'Times New Roman'})
    row, col = 4, 4
    scale = 3
    fig, axs = plt.subplots(nrows=row, ncols=col, figsize=(scale * 4 * col + 0.25 * (col-1), 4 * row + 0.3 * (row - 1)), dpi=300)
    fig.subplots_adjust(hspace=0.3, wspace=0.15)

    valid_folders = []
    for item in os.listdir("."):
        if os.path.isdir(os.path.join("./", item)) and item.startswith("case_"):
            try:
                i = int(item.split("_")[1])
                valid_folders.append((i, item))
            except (IndexError, ValueError):
                continue
    valid_folders.sort()

    # trace_names = ["Azure_code", "Azure_code", "Azure_conv", "Azure_conv"]
    # scaler_policy = ["reactive", "proactive", "hybrid", "pass", "llumnix", "preserve", "preserve_err"]
    # for _, case_dir in valid_folders:
    for row_id in range(row):
        for col_id in range(col):
            axs[row_id, col_id].grid(color='lightgray', axis="y", zorder=1, alpha=0.5)
            axs[row_id, col_id].grid(color='lightgray', axis="x", zorder=1, alpha=0.5)
    
    handles = []
    labels = []
    for idx, (case_id, case_dir) in enumerate(valid_folders):
        case_data, config = get_data(case_dir)
        if case_data is None:
            continue
        print(config["request_config"]["workload"], config["scheduler_config"]["scaler_policy"])
        # row_id = trace_names.index(config["request_config"]["workload"])
        # col_id = scaler_policy.index(config["scheduler_config"]["scaler_policy"])
        # row_id, col_id = get_row_col(idx)
        row_id, col_id = get_mapping(config["request_config"]["workload"], config["scheduler_config"]["scaler_policy"])
        print(f"row_id: {row_id}, col_id: {col_id}")
        value = plot_data(axs, row_id, col_id, case_data, config)
        if value is not None:
            h, l = value
            handles.extend(h)
            labels.extend(l)
    
    
    for ax in axs.flat:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    desired_order = ['Azure-code TPS', 'Azure-chat TPS', 'Normalized Latency', 'Instance Number']
    label_handle_dict = dict(zip(unique_labels, unique_handles))
    ordered_handles = [label_handle_dict[label] for label in desired_order if label in label_handle_dict]
    ordered_labels = [label for label in desired_order if label in label_handle_dict]
    fig.legend(ordered_handles, ordered_labels, loc='upper center', prop={'size': 33}, ncol=4, bbox_to_anchor=(0.5, 0.975))

    # fig.legend(loc='upper center', prop={'size': 30}, ncol=5, bbox_to_anchor=(0.5, 1.06))
    
    plt.savefig(f"./RQ2_instance_scaling.pdf", bbox_inches='tight')
    plt.close()

def merge_result_files(base_file, target_file, ratio=0.5):
    """
    merge two result files, replace the latency metrics of some requests in base_file with the values in target_file
    
    Args:
        base_file: base result file path
        target_file: target result file path
        ratio: the ratio of requests to replace, default 0.5
    """
    # read two files
    with open(base_file, 'r') as f:
        base_data = [json.loads(line) for line in f]
    with open(target_file, 'r') as f:
        target_data = [json.loads(line) for line in f]
    
    # calculate the time offset
    base_start_time = base_data[0]['record_time']
    target_start_time = target_data[0]['record_time']
    time_offset = base_start_time - target_start_time
    
    # create the mapping from request_id to target_data
    target_map = {item['request_id']: item for item in target_data}
    
    # determine the number of requests to replace
    num_to_replace = int(len(base_data) * ratio)
    replace_ids = set(np.random.choice(len(base_data), num_to_replace, replace=False))
    
    for i, base_item in enumerate(base_data):
        if i in replace_ids and base_item['request_id'] in target_map:
            target_item = target_map[base_item['request_id']]
            
            # replace the latency metrics
            base_item['latency'] = target_item['latency']
            base_item['TTFT'] = target_item['TTFT']
            base_item['TBPT'] = target_item['TBPT']
            
            # adjust the timestamp
            if 'itl' in target_item:
                base_item['itl'] = [t + time_offset for t in target_item['itl']]
    
    # save the merged result
    output_file = base_file.replace('.json', '_merged.json')
    with open(output_file, 'w') as f:
        for item in base_data:
            f.write(json.dumps(item) + '\n')
    
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--merge', action='store_true', help='Merge result files')
    parser.add_argument('--base', type=str, help='Base result file path')
    parser.add_argument('--target', type=str, help='Target result file path')
    parser.add_argument('--ratio', type=float, default=0.5, help='Ratio of requests to replace')
    args = parser.parse_args()
    
    if args.merge:
        if not args.base or not args.target:
            print("Error: Both base and target file paths are required for merging")
            exit(1)
        output_file = merge_result_files(args.base, args.target, args.ratio)
        print(f"Merged result saved to: {output_file}")
    else:
        plot_figures()