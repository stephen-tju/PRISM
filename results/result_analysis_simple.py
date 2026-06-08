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
        list(subploting_map["Load"].keys()) + list(subploting_map["Internal"].keys())
    }
    for line in log_lines:
        if "[INFO]" in line: continue

        # [DEBUG] [2024-12-15 20:08:38,597]: ('Load', [{'request_num': 182, 'prefill.....}])
        timestamp = re.findall(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}', line)[0]
        time = datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S,%f').timestamp()
        try:
            log_type, log_contents = eval(line[35:])  # FIX: Eval is dangerous; Based on the log format
        except:
            print(f"Error: unable to parse line: {line}")
            continue

        if instance_id >= len(log_contents):
            continue
        log_content = log_contents[instance_id]  # only 1 ploting instance
        if log_type not in subploting_map: 
            continue
        for key in subploting_map[log_type]:
            data[key]["time"].append(time)
            data[key]["value"].append(log_content.get(key, 0))
        
    ### post processing 
    ## Shift in accumulated values
    if "vllm:num_preemptions_total" in data and data["vllm:num_requests_running"]["value"]:
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

    result_timestamp = result_files[0].split("_",1)[1].split(".")[0]

    df = procecss_result_file(case_dir, result_files[-1])
    if 'TTFT' not in df.columns:
        print(f"Error: unable to find TTFT in case {case_id}, no successful finished requests")
        return 0

    with open(os.path.join(case_dir, "config.json"), "r") as file:
        config = json.load(file)
    request_config = config["request_config"]
    scheduler_config = config["scheduler_config"]

    ### Preprocessing
    # Standardize timestamp
    timestamp_base = df["record_time"].values[0]
    df["record_time"] = df["record_time"] - timestamp_base

    # Filter out failed requests
    # status=200 / expection / dummyReq
    df_latency = df.dropna(subset=['TTFT', 'latency', 'record_time', 'TBPT']).copy()

    df_latency["latency_normalized"] = df_latency["latency"] / df_latency["generated_tokens"]

    ### Ploting timeline per instance
    instance_fig_path = os.path.join(case_dir, "monitor_plot_instance_simple.png")
    # if not os.path.exists(instance_fig_path):
    if True:
        print(f"Timestamp base: {timestamp_base}")

        num_subplots = 3
        num_instances = scheduler_config["num_instances"]
        ploting_num_instances = max(2, num_instances)
        fig, axs = plt.subplots(ploting_num_instances, num_subplots, figsize=(8 * num_subplots, 3 * ploting_num_instances))

        def monitor_plot_instance(axs_id, instance_id, time, data, label):
            axs[instance_id][axs_id].plot(time, data, label=label)

        
        subploting_map = {
            "Load": {
                "incoming_prefill_tokens": 0,
            },
            "Internal": {
                "vllm:gpu_cache_usage_perc": 1,
            }
        }

        incoming_prefill_tokens_min, incoming_prefill_tokens_max = 1e9, 0
        latency_normalized_min, latency_normalized_max = 1e9, 0
        latency_normalized_smoothed_min, latency_normalized_smoothed_max = 1e9, 0

        for instance_id in range(num_instances):
            monitor_data = process_monitor_log(case_dir, subploting_map, instance_id)
            if monitor_data is None: continue

            ## 1. monitor 
            for metric, subplot_id in chain(subploting_map["Load"].items(), subploting_map["Internal"].items()):
                if metric == "incoming_prefill_tokens":
                    monitor_plot_instance(subplot_id, instance_id, 
                        monitor_data[metric]["time"] - timestamp_base, 
                        monitor_data[metric]["value"], "Incoming Prefill Tokens")
                elif metric == "vllm:gpu_cache_usage_perc":
                    monitor_data[metric]["value"] = [v * 100 for v in monitor_data[metric]["value"]]
                    monitor_plot_instance(subplot_id, instance_id, 
                        monitor_data[metric]["time"] - timestamp_base, 
                        monitor_data[metric]["value"], "GPU Memory Usage (%)")
                else:
                    pass

            incoming_prefill_tokens_min = min(incoming_prefill_tokens_min, min(monitor_data["incoming_prefill_tokens"]["value"]))
            incoming_prefill_tokens_max = max(incoming_prefill_tokens_max, max(monitor_data["incoming_prefill_tokens"]["value"]))
            


            df_latency_i = df_latency[df_latency["instance_id"] == instance_id]

            ## 2. Normalized Latency
            # # a. Imeadiate plot
            # monitor_plot_instance(2, instance_id, 
            #     df_latency_i["record_time"], df_latency_i["latency_normalized"], 
            #     "Normalized Latency") 
            # latency_normalized_min = min(latency_normalized_min, df_latency_i["latency_normalized"].min())
            # latency_normalized_max = max(latency_normalized_max, df_latency_i["latency_normalized"].max())

            # b. Smoothed 
            n = 1.0
            bins = np.floor(df_latency_i["record_time"] * n).astype(int)
            latency_normalized_smoothed_time = np.unique(bins)
            latency_normalized_smoothed_data = np.array([
                np.mean(df_latency_i["latency_normalized"][bins == b]) for b in latency_normalized_smoothed_time
            ])

            # latency_normalized_smoothed_data = np.minimum(latency_normalized_smoothed_data, 0.1)

            monitor_plot_instance(2, instance_id, 
                latency_normalized_smoothed_time.astype(float) / n, 
                latency_normalized_smoothed_data, 
                f"Normalized Latency per {1.0/n}s")
            latency_normalized_smoothed_min = min(latency_normalized_smoothed_min, latency_normalized_smoothed_data.min())
            latency_normalized_smoothed_max = max(latency_normalized_smoothed_max, latency_normalized_smoothed_data.max())


            

        ## Unify y-axis range 
        def monitor_set_ylim(axs_id, ylim_min, ylim_max):
            for i in range(num_instances):
                axs[i][axs_id].set_ylim(ylim_min, ylim_max)

        monitor_set_ylim(0, incoming_prefill_tokens_min - 1000, incoming_prefill_tokens_max + 2000)
        monitor_set_ylim(1, -5, 105)
        # monitor_set_ylim(2, latency_normalized_min, latency_normalized_max)
        monitor_set_ylim(2, 0.01, latency_normalized_smoothed_max + 0.01)


        ### Z. config and save
        for ax in axs.flat:
            if ax.lines: ax.legend(prop = {'size':12})
            ax.grid(True, which='both', axis='both')
        fig.tight_layout()
        fig.suptitle(f"Instance plot - case_{case_id} - {result_timestamp} - #Q={request_config['request_num']}, QPS={request_config['qps']}, policy={scheduler_config['scheduler_policy']}")
        fig.savefig(instance_fig_path.replace(".png", f".png"), bbox_inches="tight", dpi=200)
        plt.close(fig)
    
    
    return 1


if __name__ == "__main__":
    for i in range(197,200):
        exist = process_case(i)
        # if not exist:
        #     break
        print(i)
