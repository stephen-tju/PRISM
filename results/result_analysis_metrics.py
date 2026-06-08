import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import re
import datetime
from itertools import chain
from collections import defaultdict


def process_monitor_log(case_dir, subploting_map, instance_id=0):
    log_path = os.path.join(case_dir, "monitor.log")
    try:
        with open(log_path, "r") as file:
            log_lines = file.readlines()
    except:
        print(f"No monitor log file: {log_path}")
        return None

    ### parsing
    data = {key: {"time": [], "value": []} for key in 
        # list(subploting_map["Load"].keys()) + list(subploting_map["Internal"].keys())
        chain.from_iterable(subploting_map.values())
    }
    for line in log_lines:
        if "[INFO]" in line: continue

        # [DEBUG] [2024-12-15 20:08:38,597]: ('Load', [{'request_num': 182, 'prefill.....}])
        timestamp = re.findall(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}', line)[0]
        time = datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S,%f').timestamp()
        try:
            log_type, log_contents = eval(line[35:])  # FIX: Eval is dangerous; Based on the log format, skip timestamp
        except:
            print(f"Error: unable to parse line: {line}")
            continue

        if instance_id >= len(log_contents):
            continue
        log_content = log_contents[instance_id]  # only 1 ploting instance
        for key in subploting_map[log_type]:
            data[key]["time"].append(time)
            data[key]["value"].append(log_content.get(key, 0))
        
    ### post processing 
    ## Shift in accumulated values
    if "vllm:num_requests_running" in data and data["vllm:num_requests_running"]["value"]:
        base_value = data["vllm:num_preemptions_total"]["value"][0]
        data["vllm:num_preemptions_total"]["value"] = [
            value - base_value for value in data["vllm:num_preemptions_total"]["value"]
        ]

    return data


def procecss_result_file(case_dir, result_file):
    warm_up_request_id, warm_end_request_id = 0, 0

    result_path = os.path.join(case_dir, result_file)
    results = []
    total_requests = 0
    try:
        with open(result_path, "r") as file:
            lines = file.readlines()
    except:
        print(f"Error: unable to read result file: {result_path}")
        return 0

    total_requests = len(lines)
    for line in lines:
        record = json.loads(line.strip())
        if (record["request_id"] < warm_up_request_id or 
            record["request_id"] > total_requests - warm_end_request_id):
            continue
        results.append(record)

    return pd.DataFrame(results)


def filter_invalid_data(df, case_id):
    invalid_counts = (df.isin([np.nan, np.inf, -np.inf])).sum()
    if invalid_counts.sum() > 0:
        print(f"Case {case_id} contains invalid data:")
        print(invalid_counts[invalid_counts > 0])
    return df.replace([np.inf, -np.inf], np.nan).dropna()


def plot_cdf(data_list, metric_name, ax):
    hist, bin_edges = np.histogram(data_list, bins=100)
    cumsum = np.cumsum(hist)
    pdf_line = ax.bar(bin_edges[:-1], hist, width=bin_edges[1]-bin_edges[0], alpha=0.3, label='PDF')
    cdf_line = ax.plot(bin_edges[1:], cumsum / np.sum(hist) * 100, alpha=0.3, label="CDF")
    ax.set_title(f"{metric_name} Distribution")
    ax.set_xlabel(metric_name)
    ax.set_ylabel("Frequency")
    ax.legend()



def calculate_metrics(data_list, metric_name, metrics_dict):
    valid_data = np.array(data_list, dtype=np.float64)
    valid_data = valid_data[np.isfinite(valid_data)] 

    if valid_data.size == 0:
        print(f"Warning: No valid data for {metric_name}")
        metrics_dict[metric_name] = {
            "mean": None, "std": None, "p50": None, "p80": None, 
            "p90": None, "p95": None, "p99": None, "p999": None
        }
        return

    metric_json = {
        "mean": np.mean(valid_data),
        "std": np.std(valid_data),
        "p50": np.percentile(valid_data, 50),
        "p80": np.percentile(valid_data, 80),
        "p90": np.percentile(valid_data, 90),
        "p95": np.percentile(valid_data, 95),
        "p99": np.percentile(valid_data, 99),
        "p999": np.percentile(valid_data, 99.9),
    }
    metrics_dict[metric_name] = metric_json


# Percentage of SLO violated requests
def calculate_metrics_SLO_violation(data_list, metric_name, metrics_dict, slo_threshold):
    valid_data = np.array(data_list, dtype=np.float64)
    valid_data = valid_data[np.isfinite(valid_data)] 

    if valid_data.size == 0:
        metrics_dict[metric_name]["slo_violation_perc"] = None
        return
    
    slo_violation = sum([1 for data in data_list if data > slo_threshold])
    slo_violation_perc = slo_violation / len(data_list) * 100
    metrics_dict[metric_name]["slo_violation_perc"] = slo_violation_perc


def process_case(case_id):
    case_dir = f"./cases/case_{case_id}"
    if not os.path.exists(case_dir):
        return 0

    ### A. Read results
    files = os.listdir(case_dir)
    result_files = [f for f in files if f.startswith("result") and f.endswith(".json")]
    if len(result_files) != 1: 
        print(f"Error: found {len(result_files)} result file in case {case_id}")
        return 0
    result_timestamp = result_files[-1].split("_")[1].split(".")[0]

    df = procecss_result_file(case_dir, result_files[-1])
    if 'TTFT' not in df.columns:
        print(f"Error: unable to find TTFT in case {case_id}, no successful finished requests")
        return 0

    # df = filter_invalid_data(df, case_id)

    with open(os.path.join(case_dir, "config.json"), "r") as file:
        config = json.load(file)
    request_config = config["request_config"]
    scheduler_config = config["scheduler_config"]

    ### A+. Preprocessing
    # Standardize timestamp
    timestamp_base = df["record_time"].values[0]
    df["record_time"] = df["record_time"] - timestamp_base

    # Filter out failed requests
    df_latency = df.dropna(subset=['TTFT', 'latency', 'record_time', 'TBPT']).copy()
    # status=200 / expection / dummyReq

    # Secondary metrics
    request_per_second = defaultdict(int)
    for record_time in df_latency["record_time"]:
        second = int(record_time)
        request_per_second[second] += 1
    rps_list = list(request_per_second.values())

    df_latency["frontend_overhead"] = df_latency["frontend_latency"] - df_latency["latency"]

    df_latency.loc[:, "norm_latency"] = df_latency["latency"] / df_latency["generated_tokens"]

    ### B. metrics.json
    metrics_path = os.path.join(case_dir, "metrics.json")
    # if not os.path.exists(metrics_path):
    if True:
        metrics_dict = {}
        # distribution
        calculate_metrics(list(df_latency["TTFT"]), "TTFT", metrics_dict)
        calculate_metrics(list(df_latency["TBPT"]), "TBT", metrics_dict)
        calculate_metrics(list(df_latency["latency"]), "Latency", metrics_dict)
        calculate_metrics(list(df_latency["norm_latency"]), "Norm Latency", metrics_dict)
        calculate_metrics(rps_list, "RPS", metrics_dict)

        calculate_metrics_SLO_violation(list(df_latency["TTFT"]), "TTFT", metrics_dict, 2)
        calculate_metrics_SLO_violation(list(df_latency["TBPT"]), "TBT", metrics_dict, 0.1)
        calculate_metrics_SLO_violation(list(df_latency["latency"]), "Latency", metrics_dict, 80)
        calculate_metrics_SLO_violation(list(df_latency["norm_latency"]), "Norm Latency", metrics_dict, 0.25)

        # overhead
        if "frontend_latency" in df_latency.columns:
            calculate_metrics(list(df_latency["frontend_overhead"]), "Frontend Overhead", metrics_dict)
        if "req_predictor_time" in df_latency.columns:
            calculate_metrics(list(df_latency["req_predictor_time"]), "Request Predictor Overhead", metrics_dict)
        
        with open(metrics_path, "w") as file:
            json.dump([request_config, scheduler_config, metrics_dict], file, indent=4)

        print("mean,P50,P90 of metric TTFT: {:.3f} {:.3f} {:.3f}".format(
            metrics_dict["TTFT"]["mean"], metrics_dict["TTFT"]["p50"], metrics_dict["TTFT"]["p90"]))
        print("mean,P50,P90 of metric TBT: {:.3f} {:.3f} {:.3f}".format(
            metrics_dict["TBT"]["mean"], metrics_dict["TBT"]["p50"], metrics_dict["TBT"]["p90"]))

    # CDF
    cdf_fig_path = os.path.join(case_dir, "CDFs.png")
    if not os.path.exists(cdf_fig_path):
    # if True:
        cdf_datas = [df_latency["TTFT"], df_latency["latency"], df_latency["TBPT"], rps_list]
        cdf_names = ["TTFT", "Latency", "TBPT", "RPS"]

        if "frontend_latency" in df_latency.columns:   
            cdf_datas.append(df_latency["frontend_overhead"])
            cdf_names.append("frontend_overhead")
        if "req_predictor_time" in df_latency.columns:
            cdf_datas.append(df_latency["req_predictor_time"])
            cdf_names.append("req_predictor_time")

        cdf_fig, cdf_axs = plt.subplots(len(cdf_names), 1, figsize=(6, 3 * len(cdf_names)))    
        for ax, data, name in zip(cdf_axs, cdf_datas, cdf_names):
            plot_cdf(data, name, ax)
        
        plt.tight_layout()
        plt.savefig(cdf_fig_path, bbox_inches="tight", dpi=300)
        plt.close()
    
    return 1


if __name__ == "__main__":
    for i in range(0,500):
        exist = process_case(i)
        # if not exist:
        #     break
        print(i)
