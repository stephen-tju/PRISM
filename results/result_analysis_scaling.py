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
import math

'''
Scaling senario: Running instances are always left aligned, scaling happens only at the right side
'''
MAX_NUM_INSTANCES = 8


# def process_monitor_log(case_dir, subploting_map, instance_id=0):
#     log_path = os.path.join(case_dir, "monitor.log")
#     try:
#         with open(log_path, "r") as file:
#             log_lines = file.readlines()
#     except:
#         print(f"No monitor log file: {log_path}")
#         return None

#     ### parsing
#     data = {key: {"time": [], "value": []} for key in 
#         # list(subploting_map["Load"].keys()) + list(subploting_map["Internal"].keys())
#         chain.from_iterable(subploting_map.values())
#     }
#     for line in log_lines:
#         if "[INFO]" in line: continue

#         # [DEBUG] [2024-12-15 20:08:38,597]: ('Load', [{'request_num': 182, 'prefill.....}])
#         timestamp = re.findall(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}', line)[0]
#         time = datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S,%f').timestamp()
#         try:
#             log_type, log_contents = eval(line[35:])  # TODO: Eval is dangerous; Based on the log format, skip timestamp
#         except:
#             print(f"Error: unable to parse line: {line}")
#             continue

#         if instance_id >= len(log_contents):
#             continue  # SCALABLE

#         log_content = log_contents[instance_id]  # only 1 ploting instance
#         for key in subploting_map[log_type]:
#             data[key]["time"].append(time)
#             data[key]["value"].append(log_content.get(key, 0))
        
#     ### post processing 
#     ## Shift in accumulated values
#     if "vllm:num_requests_running" in data and data["vllm:num_requests_running"]["value"]:
#         base_value = data["vllm:num_preemptions_total"]["value"][0]
#         data["vllm:num_preemptions_total"]["value"] = [
#             value - base_value for value in data["vllm:num_preemptions_total"]["value"]
#         ]

#     return data


# New: process all instances
def process_monitor_log_all(case_dir, all_metric_keys):
    log_path = os.path.join(case_dir, "monitor.log")
    try:
        with open(log_path, "r") as file:
            log_lines = file.readlines()
    except:
        print(f"No monitor log file: {log_path}")
        return None

    '''
        [inst_0, inst_1, ...]
         inst_0: {
            key_0: {"time": [], "value": []}, 
            key_1: {"time": [], "value": []}, ...}
    '''
    all_data = [ {key: {"time": [], "value": []} for key in all_metric_keys} for _ in range(MAX_NUM_INSTANCES) ]
    
    num_instances_data = {"time": [], "value": []}

    for line in log_lines:
        # [DEBUG] [2024-12-15 20:08:38,597]: ('Load', [{'request_num': 182, 'prefill.....}])
        if "[INFO]" in line: continue
        if "[ERROR]" in line: continue
        timestamp = re.findall(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}', line)[0]
        time = datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S,%f').timestamp()
        try:
            log_type, log_contents = eval(line[35:])  # TODO: Eval is dangerous; Based on the log format, skip timestamp
        except:
            print(f"Error: unable to parse line: {line}")
            continue

        current_num_instances = len(log_contents)
        num_instances_data["time"].append(time)
        num_instances_data["value"].append(current_num_instances)

        for instance_id, log_content in enumerate(log_contents):
            for key in all_metric_keys:
                if key in log_content:
                    all_data[instance_id][key]["time"].append(time)
                    all_data[instance_id][key]["value"].append(log_content[key])

    # Post processing
    for instance_id in range(MAX_NUM_INSTANCES):
        if "vllm:num_requests_running" in all_data[instance_id] and len(all_data[instance_id]["vllm:num_requests_running"]["value"]) > 0:
            base_value = all_data[instance_id]["vllm:num_preemptions_total"]["value"][0]
            all_data[instance_id]["vllm:num_preemptions_total"]["value"] = [
                value - base_value for value in all_data[instance_id]["vllm:num_preemptions_total"]["value"]
            ]
    
    return all_data, num_instances_data


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


def plot_cdf(data_list, metric_name, ax):
    hist, bin_edges = np.histogram(data_list, bins=50)
    cumsum = np.cumsum(hist)
    pdf_line = ax.bar(bin_edges[:-1], hist, width=bin_edges[1]-bin_edges[0], alpha=0.3, label='PDF')
    cdf_line = ax.plot(bin_edges[1:], cumsum / np.sum(hist) * 100, alpha=0.3, label="CDF")
    ax.set_title(f"{metric_name} Distribution")
    ax.set_xlabel(metric_name)
    ax.set_ylabel("Frequency")
    ax.legend()


def plot_violinplot(data_list, metric_names, case_dir, request_config, instance_num):
    # request_num, qps, input_len_mean, input_len_range, output_len_mean, output_len_range = config_list
    request_num = request_config["request_num"]
    qps = request_config["qps"]
    input_len_mean = request_config["random_prompt_lens_mean"]
    input_len_range = request_config["random_prompt_lens_range"]
    output_len_mean = request_config["random_response_lens_mean"]
    output_len_range = request_config["random_response_lens_range"]

    colors = ["#428B2C", "#F79B25", "#F79B25", "#4e75b0"]

    # Creating a figure and adding the config text at the top
    fig, axs = plt.subplots(1, 4, figsize=(12, 6))
    config_text = (
        f"Config:\n"
        f"Request: {request_num}, RPS: {qps}, "
        # f"Input Length {input_len_mean} +- {input_len_range}, "
        # f"Output Length: {output_len_mean} +- {output_len_range}"
        f"Instance Number: {instance_num}"
    )
    fig.text(
        0.5, 0.95, config_text, ha="center", fontsize=12, fontweight="bold", wrap=True
    )

    for i, data in enumerate(data_list):
        p99_value = np.percentile(data, 99)

        # Creating the violin plot for each metric
        parts = axs[i].violinplot([data], showmeans=False, showmedians=True)

        # Customizing the color and style of the violin plot
        parts["bodies"][0].set_facecolor(colors[i])
        parts["bodies"][0].set_edgecolor("black")
        parts["bodies"][0].set_alpha(0.8)

        # Adding P99 value text and line
        axs[i].text(
            1,
            max(data) * 1.05,
            f"P99: {p99_value:.2f}",
            ha="center",
            color=colors[i],
            fontsize=10,
            fontweight="bold",
        )
        axs[i].axhline(p99_value, color="red", linestyle="--", linewidth=1)

        # Setting axis labels and limits
        axs[i].set_xlabel(f"{metric_names[i]}")
        axs[i].set_xticks([1])
        y_min = min(data) * 0.9
        y_max = max(data) * 1.1
        axs[i].set_ylim(y_min, y_max)

    # Adjusting layout to fit the config text at the top
    plt.tight_layout(
        rect=[0, 0, 1, 0.93]
    )  # Reserve space for the config text at the top
    figure_path = os.path.join(case_dir, f"violin_plot.png")
    plt.savefig(figure_path, bbox_inches="tight")
    plt.close()


def calculate_metrics(data_list, metric_name, metrics_dict):
    mean_val = np.mean(data_list)
    std_val = np.std(data_list)
    p50 = np.percentile(data_list, 50)
    p80 = np.percentile(data_list, 80)
    p90 = np.percentile(data_list, 90)
    p95 = np.percentile(data_list, 95)
    p99 = np.percentile(data_list, 99)
    p999 = np.percentile(data_list, 99.9)
    p9999 = np.percentile(data_list, 99.99)

    metric_json = {
        "mean": mean_val,
        "std": std_val,
        "p50": p50,
        "p80": p80,
        "p90": p90,
        "p95": p95,
        "p99": p99,
        "p999": p999,
        # "p9999": p9999,
    }
    metrics_dict[metric_name] = metric_json


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

    with open(os.path.join(case_dir, "config.json"), "r") as file:
        config = json.load(file)
    request_config = config["request_config"]
    scheduler_config = config["scheduler_config"]
    
    workload = request_config["workload"]
    assert workload == "Azure_code_peak" or workload == "Azure_conv_peak"

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

    if "frontend_latency" in df_latency.columns:
        df_latency["frontend_overhead"] = df_latency["frontend_latency"] - df_latency["latency"]


    ### B. metrics.json
    metrics_path = os.path.join(case_dir, "metrics.json")
    while not os.path.exists(metrics_path):
        metrics_dict = {}
        # distribution
        calculate_metrics(list(df_latency["TTFT"]), "TTFT", metrics_dict)
        calculate_metrics(list(df_latency["latency"]), "Latency", metrics_dict)
        calculate_metrics(list(df_latency["TBPT"]), "TBPT", metrics_dict)
        calculate_metrics(rps_list, "RPS", metrics_dict)
        # overhead
        if "frontend_latency" in df_latency.columns:
            calculate_metrics(list(df_latency["frontend_overhead"]), "Frontend Overhead", metrics_dict)
        
        with open(metrics_path, "w") as file:
            json.dump([request_config, scheduler_config, metrics_dict], file, indent=4)

        print("mean,P50,P90 of metric TTFT: {:.3f} {:.3f} {:.3f}".format(
            metrics_dict["TTFT"]["mean"], metrics_dict["TTFT"]["p50"], metrics_dict["TTFT"]["p90"]))
        print("mean,P50,P90 of metric TBPT: {:.3f} {:.3f} {:.3f}".format(
            metrics_dict["TBPT"]["mean"], metrics_dict["TBPT"]["p50"], metrics_dict["TBPT"]["p90"]))

    ### C. ploting data distribution
    # # Violin
    # plot_violinplot(
    #     [df_latency["TTFT"], df_latency["latency"], df_latency["TBPT"], rps_list],
    #     ["TTFT", "Latency", "TBPT", "RPS"],
    #     case_dir,
    #     request_config,
    #     scheduler_config["num_instances"]
    # )

    # CDF
    cdf_fig_path = os.path.join(case_dir, "CDFs.png")
    while not os.path.exists(cdf_fig_path):
        cdf_datas = [df_latency["TTFT"], df_latency["latency"], df_latency["TBPT"], rps_list]
        cdf_names = ["TTFT", "Latency", "TBPT", "RPS"]
        if "frontend_latency" in df_latency.columns:
            cdf_datas.append(df_latency["frontend_overhead"])
            cdf_names.append("frontend_overhead")

        cdf_fig, cdf_axs = plt.subplots(len(cdf_names), 1, figsize=(6, 3 * len(cdf_names)))    
        for ax, data, name in zip(cdf_axs, cdf_datas, cdf_names):
            plot_cdf(data, name, ax)
        
        plt.tight_layout()
        plt.savefig(cdf_fig_path, bbox_inches="tight", dpi=300)
        plt.close()




    # GPU_max_tokens = 16 * 1741  # A100-40GB, 0.5util, Llama-3-8b
    # GPU_max_tokens = 16 * 1858  # A100-40GB, 0.7util, Llama-2-7b
    # GPU_max_tokens = 16 * 1707  # A100-40GB, 0.67util, Llama-2-7b
    # GPU_max_tokens = 16 * 2872  # A100-40GB, DEFAULT util, Llama-2-7b
    # GPU_max_tokens = 16 * 1108  # A100-40GB, 0.55 util, Llama-2-7b
    GPU_max_tokens = 16 * 2306  # A40-48GB, 0.7 util, Llama-2-7b

    subploting_map = {
        "Load": {
            "request_num": 0,
            "incoming_prefill_tokens": 1,
            "incoming_decode_tokens": 1,
            # "incoming_all_tokens": 1,
            "expected_token_usage": 1,
            "current_token_usage": 1,
            "lookahead_max_tokens": 1,
        },
        "Internal": {
            "vllm:num_requests_running": 0,
            "vllm:num_requests_waiting": 0,
            "vllm:num_preemptions_total": 0,
            "vllm:gpu_cache_usage_perc": 1,
        },
        # "External": {
        #     "gpu_utilization": 2
        # }
    }
    # ["request_num", "incoming_prefill_tokens", "incoming_decode_tokens", ...]
    all_metric_keys = list(chain.from_iterable(subploting_map.values()))  
    # [0, 1, 1, ...]
    all_metric_subploting_ids = [value for sublist in subploting_map.values() for value in sublist.values()]


    # ### D. ploting single timeline
    # monitor_fig_path = os.path.join(case_dir, "monitor_plot.png")
    # while not os.path.exists(monitor_fig_path): # Ploting is slow, skip if already exists
    #     monitor_data = process_monitor_log(case_dir, subploting_map)
    #     if monitor_data is None: # No monitor log
    #         break

    #     print(f"Timestamp base: {timestamp_base}")

    #     monitor_data["vllm:gpu_cache_usage_perc"]["value"] = [
    #         value * GPU_max_tokens for value in monitor_data["vllm:gpu_cache_usage_perc"]["value"]
    #     ]

    #     ## D.a ploting monitor data
    #     fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    #     for metric, subplot_id in chain(subploting_map["Load"].items(), subploting_map["Internal"].items()):
    #         i, j = subplot_id // 3, subplot_id % 3
    #         axs[i][j].plot(monitor_data[metric]["time"] - timestamp_base, 
    #                     monitor_data[metric]["value"], label=metric)
        
    #     # Extra: ploting failed request
    #     df_failed = df[~df.index.isin(df_latency.index)].copy()
    #     if df_failed.shape[0] > 0:
    #         df_failed.loc[:, "failed"] = 1
    #         df_failed.loc[:, "failed"] = df_failed.loc[:, "failed"].cumsum()
    #         axs[0][0].plot(df_failed["record_time"], 
    #                     df_failed["failed"], label="failed_total")

    #     # # Extra: ploting aborted request
    #     # df_aborted = df[df["generated_tokens"] < df["expected_tokens"]]
    #     # if df_aborted.shape[0] > 0:
    #     #     df_aborted.loc[:, "aborted"] = 1
    #     #     df_aborted.loc[:, "aborted"] = df_aborted.loc[:, "aborted"].cumsum()
    #     #     axs[0][0].plot(df_aborted["record_time"], 
    #     #                 df_aborted["aborted"], label="aborted_total")

    #     # Extra: ploting request latency
    #     axs[1][0].plot(df_latency["record_time"], df_latency["TTFT"], label="TTFT naive")
    #     axs[1][1].plot(df_latency["record_time"], df_latency["TBPT"], label="TBPT naive")

    # ''' smoothed: average per 1/n seconds '''
    # def smoothed_by_1n_seconds(n, timestamps, data):
    #     bins = np.floor(timestamps * n).astype(int) 
    #     unique_bins = np.unique(bins)
    #     smoothed_scaled_timestamp = unique_bins #+ 0.5 / n
    #     smoothed_data = np.array([np.mean(data[bins == b]) for b in unique_bins])
    #     return smoothed_scaled_timestamp, smoothed_data

    # ! aggregated_timestamps might have empty values
    def aggregate_by_interval(timestamps, datas, interval, method='mean'):
        bins = np.floor(timestamps / interval).astype(int)
        
        df = pd.DataFrame({'timestamp': bins, 'datas': datas})
        aggregated_data = df.groupby('timestamp').agg({'datas': method})

        aggregated_timestamps = aggregated_data.index * interval
        aggregated_data = aggregated_data.values
        return aggregated_timestamps, aggregated_data


    #     ## D.b. request 
    #     ## 1. TTFT
    df_latency["ttft_timestamp"] = df_latency["itl"].apply(
        lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None
    )
    df_latency["ttft_timestamp"] = df_latency["ttft_timestamp"] - timestamp_base
    #     df_latency = df_latency.sort_values(by="ttft_timestamp")
    #     axs[1][0].plot(df_latency["ttft_timestamp"], 
    #                     df_latency["TTFT"], 
    #                     label="TTFT Imd.")
    #     # TTFT feature
    #     ttft_mean = df_latency["TTFT"].mean()
    #     ttft_std = df_latency["TTFT"].std()


    #     ## 2. Normalized TTFT
    df_latency["TTFT_normalized"] = df_latency["TTFT"] / df_latency["prompt_tokens"]
    #     axs[1][0].plot(df_latency["ttft_timestamp"],
    #                         df_latency["TTFT_normalized"], 
    #                         label="normTTFT Imd.")
    #     normalized_ttft_mean = df_latency["TTFT_normalized"].mean()
    #     normalized_ttft_std = df_latency["TTFT_normalized"].std()

    #     # Normalized TTFT - average per 1/n seconds
    #     ttft_n = 2.0
    #     smoothed_timestamp_scaled, smoothed_data = smoothed_by_1n_seconds(ttft_n, df_latency["ttft_timestamp"], df_latency["TTFT_normalized"])
    #     axs[1][0].plot(smoothed_timestamp_scaled.astype(float) / ttft_n, smoothed_data, label=f"normTTFT per {1.0 / ttft_n}s")

    #     ## 3. TBT
    #     # TBT itl all - average itertion time per 1/n seconds
    #     timestamps_diffs = df_latency["itl"].apply(
    #         lambda x: (x[1:] - timestamp_base, np.diff(x)) if isinstance(x, list) and len(x) > 1 else ([], [])
    #     )
    #     timestamps_diffs = timestamps_diffs.reset_index(drop=True)   
    #     tbt_timestamp = np.concatenate(timestamps_diffs.map(lambda x: x[0]))
    #     tbt_data = np.concatenate(timestamps_diffs.map(lambda x: x[1]))

    #     tbt_n = 2.0
    #     smoothed_timestamp_scaled, smoothed_data = smoothed_by_1n_seconds(tbt_n, tbt_timestamp, tbt_data)
    #     axs[1][1].plot(smoothed_timestamp_scaled.astype(float) / tbt_n, smoothed_data, label=f"TBPT per {1.0 / tbt_n}s")
    #     # TBT feature
    #     tbpt_mean = smoothed_data.mean()
    #     tbpt_std = smoothed_data.std()

    #     ## 4. Latency
    #     df_latency["latency_mid_timestamp"] = df_latency["record_time"] - df_latency["latency"] / 2
    df_latency["latency_normalized"] = df_latency["latency"] / df_latency["generated_tokens"]
    #     # TODO ploting all instance

    #     ## D.c. config and save
    #     for ax in axs.flat:
    #         if ax.lines: ax.legend()
    #         ax.grid(True, which='both', axis='both')
    #     fig.tight_layout()
    #     fig.savefig(monitor_fig_path, bbox_inches="tight", dpi=300)
    #     plt.close(fig)



    ### E. ploting timeline per instance
    instance_fig_path = os.path.join(case_dir, "monitor_plot_instance.png")
    if not os.path.exists(instance_fig_path):
    # if True:
        # Whole plot
        h = MAX_NUM_INSTANCES + 2  # +1 for all instance-lines ploting together, +2 for total-line ploting
        num_subplots = 6
        fig, axs = plt.subplots(h, num_subplots, figsize=(10 * num_subplots, 3 * (h)))
        # Split plot for smaller figure
        num_subplots_0 = 3
        fig_0, axs_0 = plt.subplots(h, num_subplots_0, figsize=(8 * num_subplots_0, 3 * (h)))
        num_subplots_1 = num_subplots - num_subplots_0
        fig_1, axs_1 = plt.subplots(h, num_subplots_1, figsize=(5 * num_subplots_1, 3 * (h)))

        def monitor_plot_instance(axs_id, instance_id, time, data, label):
            if (axs_id < num_subplots_0):
                axs_0[instance_id][axs_id].plot(time, data, label=label)
            else:
                axs_1[instance_id][axs_id - num_subplots_0].plot(time, data, label=label)
            axs[instance_id][axs_id].plot(time, data, label=label)


        all_monitor_data, instance_nums_data = process_monitor_log_all(case_dir, all_metric_keys) 
        timestamp_end = all_monitor_data[0]["vllm:num_requests_running"]["time"][-1] - timestamp_base

        ### E.A. per instance   
        for instance_id in range(MAX_NUM_INSTANCES):
            monitor_data = all_monitor_data[instance_id]
            print(f"Instance {instance_id}: {len(monitor_data['vllm:num_requests_running']['value'])} monitor records")
            if monitor_data is None or len(monitor_data["vllm:num_requests_running"]["value"]) == 0:
                continue

            if "vllm:gpu_cache_usage_perc" in monitor_data:
                monitor_data["vllm:gpu_cache_usage_perc"]["value"] = [
                    value * GPU_max_tokens for value in monitor_data["vllm:gpu_cache_usage_perc"]["value"]
                ]

            ## a. monitor (subplot 0 - 0,1)
            for metric, subplot_id in zip(all_metric_keys, all_metric_subploting_ids):
                monitor_plot_instance(subplot_id, instance_id, 
                    monitor_data[metric]["time"] - timestamp_base,
                    monitor_data[metric]["value"], 
                    metric)
                
            ## b. req 
            ## 1. TTFT
            df_latency_i = df_latency[df_latency["instance_id"] == instance_id]
            # Immediate plot
            df_ttft = df_latency_i[["ttft_timestamp", "TTFT"]].copy()
            df_ttft_sort = df_ttft.sort_values(by="ttft_timestamp")
            monitor_plot_instance(2, instance_id, df_ttft_sort["ttft_timestamp"], df_ttft_sort["TTFT"], "TTFT Imd.")
            # monitor_plot_instance(2, MAX_NUM_INSTANCES, df_latency_i["ttft_timestamp"], df_latency_i["TTFT"], f"TTFT Imd. - {instance_id}")

            # Agg plot
            ttft_interval = 5.0
            ttft_agg_timestamp_i, ttft_agg_data_i = aggregate_by_interval(
                df_latency_i["ttft_timestamp"], df_latency_i["TTFT"], ttft_interval, method='mean')
            monitor_plot_instance(2, instance_id, ttft_agg_timestamp_i, ttft_agg_data_i, f"TTFT per {ttft_interval}s")
            monitor_plot_instance(2, MAX_NUM_INSTANCES, ttft_agg_timestamp_i, ttft_agg_data_i, f"TTFT per {ttft_interval}s - {instance_id}")


            # ## 2. Normalized TTFT
            # monitor_plot_instance(5, instance_id, df_latency_i["ttft_timestamp"], df_latency_i["TTFT_normalized"], "normTTFT Imd.")

            # # Normalized TTFT - average per 1/n seconds
            # ttft_n = 2.0
            # ttft_smoothed_scaled_timestamp_i, ttft_smoothed_data_i = smoothed_by_1n_seconds(
            #     ttft_n, df_latency_i["ttft_timestamp"], df_latency_i["TTFT_normalized"])
            # ttft_smoothed_timestamp_i = ttft_smoothed_scaled_timestamp_i.astype(float) / ttft_n
            # monitor_plot_instance(5, instance_id, ttft_smoothed_timestamp_i, ttft_smoothed_data_i, f"normTTFT per {1.0 / ttft_n}s")
            # monitor_plot_instance(5, num_instances, ttft_smoothed_timestamp_i, ttft_smoothed_data_i, f"normTTFT per {1.0 / ttft_n}s - {instance_id}")


            # ## 3. TBT
            # # TBT itl all - average itertion time per 1/n seconds
            # itl_timestamps_diffs_i = df_latency_i["itl"].apply(
            #     lambda x: (x[1:] - timestamp_base, np.diff(x)) if isinstance(x, list) and len(x) > 1 else ([], [])
            # )
            # if itl_timestamps_diffs_i.shape[0] == 0: 
            #     print(f"Error: `itl_timestamps_diffs_i` is empty for instance {instance_id}")
            #     continue
            # tbt_timestamp_i = np.concatenate([x[0] for x in itl_timestamps_diffs_i])
            # tbt_data_i = np.concatenate([x[1] for x in itl_timestamps_diffs_i])

            # # # Imediatelt plot
            # # sorted_indices = np.argsort(tbt_timestamp_i) # sort by timestamp
            # # tbt_timestamp_i = tbt_timestamp_i[sorted_indices]
            # # tbt_data_i = tbt_data_i[sorted_indices]
            # # monitor_plot_instance(3, instance_id, tbt_timestamp_i, tbt_data_i, "TBPT Imd.")

            # # smoothed: average itertion time per 1/n seconds
            # tbt_n = 2.0
            # tbt_smoothed_scaled_timestamp_i, tbt_smoothed_data_i = smoothed_by_1n_seconds(
            #     tbt_n, tbt_timestamp_i, tbt_data_i)
            # tbt_smoothed_timestamp_i = tbt_smoothed_scaled_timestamp_i.astype(float) / tbt_n
            # monitor_plot_instance(3, instance_id, tbt_smoothed_timestamp_i, tbt_smoothed_data_i, f"TBPT per {1.0 / tbt_n}s")
            # monitor_plot_instance(3, num_instances, tbt_smoothed_timestamp_i, tbt_smoothed_data_i, f"TBPT per {1.0 / tbt_n}s - {instance_id}")

            
            # ## 4. Throughput (TPS)
            # tps_n = 2.0

            # # 4.1. TTFT tokens per 1/n second 
            # bins_i = np.floor(df_latency_i["ttft_timestamp"] * tps_n).astype(int)
            # unique_bins_i = np.unique(bins_i)
            # ttft_smoothed_scaled_timestamp_i = unique_bins_i #+ 0.5 / tps_n
            # ttft_smoothed_timestamp_i = ttft_smoothed_scaled_timestamp_i.astype(float) / tps_n
            # input_tokens_per_bin = np.array([np.sum(df_latency_i["prompt_tokens"][bins_i == b]) for b in unique_bins_i])

            # # 4.2 TBT tokens per 1/n second
            # bins_i = np.floor(tbt_timestamp_i * tps_n).astype(int)
            # unique_bins_i = np.unique(bins_i)
            # tbt_smoothed_scaled_timestamp_i = unique_bins_i
            # tbt_smoothed_timestamp_i = tbt_smoothed_scaled_timestamp_i.astype(float) / tps_n
            # output_tokens_per_bin = np.array([np.sum(bins_i == b) for b in unique_bins_i])

            # # 4.3 All tokens per 1/n second
            # # scaled: int bins
            # all_smoothed_scaled_timestamp_i = np.unique(np.concatenate([ttft_smoothed_scaled_timestamp_i, tbt_smoothed_scaled_timestamp_i]))
            # all_smoothed_timestamp_i = all_smoothed_scaled_timestamp_i.astype(float) / tps_n
            # # add up input & output by smoothed_timestamp
            # aligned_input_tokens_per_bin = np.array([np.sum(input_tokens_per_bin[ttft_smoothed_scaled_timestamp_i == b]) for b in all_smoothed_scaled_timestamp_i])
            # aligned_output_tokens_per_bin = np.array([np.sum(output_tokens_per_bin[tbt_smoothed_scaled_timestamp_i == b]) for b in all_smoothed_scaled_timestamp_i])
            # all_tokens_per_bin = aligned_input_tokens_per_bin + aligned_output_tokens_per_bin

            # monitor_plot_instance(6, instance_id, all_smoothed_timestamp_i, all_tokens_per_bin, f"All TPS")
            # monitor_plot_instance(6, instance_id, ttft_smoothed_timestamp_i, input_tokens_per_bin, f"I TPS")
            # monitor_plot_instance(6, instance_id, tbt_smoothed_timestamp_i, output_tokens_per_bin, f"O TPS")

            # tps_mean = all_tokens_per_bin.mean()
            # tps_std = all_tokens_per_bin.std()


            # ## 5. PD TPS ratio: input_tokens_per_bin : output_tokens_per_bin
            # valid_mask = aligned_output_tokens_per_bin != 0
            # pd_ratio_per_bin = np.zeros_like(aligned_output_tokens_per_bin, dtype=float)
            # pd_ratio_per_bin[valid_mask] = aligned_input_tokens_per_bin.astype(float)[valid_mask] / aligned_output_tokens_per_bin.astype(float)[valid_mask]
            # max_ratio = np.max(all_tokens_per_bin[valid_mask])
            # pd_ratio_per_bin[~valid_mask] = max_ratio * 1.5

            # pd_ratio_n = 0.5
            # pd_ratio_smoothed_scaled_timestamp_i, pd_ratio_smoothed_data_i = smoothed_by_1n_seconds(
            #     pd_ratio_n, all_smoothed_timestamp_i, pd_ratio_per_bin)
            # pd_ratio_smoothed_timestamp_i = pd_ratio_smoothed_scaled_timestamp_i.astype(float) / pd_ratio_n
            # monitor_plot_instance(2, instance_id, pd_ratio_smoothed_timestamp_i, pd_ratio_smoothed_data_i, f"PD TPS Ratio")
            # monitor_plot_instance(2, num_instances, pd_ratio_smoothed_timestamp_i, pd_ratio_smoothed_data_i, f"PD TPS Ratio - {instance_id}")

            # pd_ratio_mean += pd_ratio_per_bin.mean()
            # pd_ratio_std += pd_ratio_per_bin.std()
            

            ## 6. RPS
            # Agg + Ending timestamp plot
            rps_interval = 5.0
            rps_timestamp_i, rps_data_i = aggregate_by_interval(
                df_latency_i["record_time"], df_latency_i["record_time"], 
                rps_interval, method='count')
            monitor_plot_instance(4, instance_id, rps_timestamp_i, rps_data_i, f"RPS Imd.")
            monitor_plot_instance(4, MAX_NUM_INSTANCES, rps_timestamp_i, rps_data_i, f"RPS Imd. - {instance_id}")           

            # # Mid-way timestamp plot
            # rps_n = 1.0
            # # bins_i = np.floor(df_latency_i["record_time"] * rps_n).astype(int)
            # bins_i = np.floor(df_latency_i["latency_mid_timestamp"] * rps_n).astype(int)
            # unique_bins_i = np.unique(bins_i)
            # rps_smoothed_scaled_timestamp_i = unique_bins_i
            # rps_smoothed_data_i = np.array([np.sum(bins_i == b) for b in unique_bins_i])
            # rps_smoothed_timestamp_i = rps_smoothed_scaled_timestamp_i.astype(float) / rps_n
            # monitor_plot_instance(7, instance_id, 
            #     rps_smoothed_timestamp_i, rps_smoothed_data_i, f"RPS")


            # ## 7. Latency
            # # Immediate plot, but using mid-way timestamp
            # latency_mid_timestamp = np.array(df_latency_i["latency_mid_timestamp"].tolist())
            # latency_data = np.array(df_latency_i["latency"].tolist())
            # sorted_indices = np.argsort(latency_mid_timestamp)
            # monitor_plot_instance(8, instance_id,  
            #     latency_mid_timestamp[sorted_indices], latency_data[sorted_indices], "Latency at Mid-way time")
            
            # # smoothed
            # latency_n = 1.0
            # latency_smoothed_scaled_timestamp_i, latency_smoothed_data_i = smoothed_by_1n_seconds(
            #     latency_n, df_latency_i["latency_mid_timestamp"], df_latency_i["latency"])
            # latency_smoothed_timestamp_i = latency_smoothed_scaled_timestamp_i.astype(float) / latency_n
            # monitor_plot_instance(8, instance_id, latency_smoothed_timestamp_i, latency_smoothed_data_i, f"Latency per {1.0 / latency_n}s")


            # 8. Normalized Latency
            # Immediate + ending timestamp plot
            monitor_plot_instance(3, instance_id, df_latency_i["record_time"], df_latency_i["latency_normalized"], "Norm. Latency Imd.")

            # Agg + ending timestamp plot
            norm_latency_interval = 5.0
            norm_latency_agg_timestamp_i, norm_latency_agg_data_i = aggregate_by_interval(
                df_latency_i["record_time"], df_latency_i["latency_normalized"], norm_latency_interval, method='mean')
            monitor_plot_instance(3, instance_id, norm_latency_agg_timestamp_i, norm_latency_agg_data_i, f"Norm. Latency per {norm_latency_interval}s")
            monitor_plot_instance(3, MAX_NUM_INSTANCES, norm_latency_agg_timestamp_i, norm_latency_agg_data_i, f"Norm. Latency per {norm_latency_interval}s - {instance_id}")

            # # Imediate + Mid-way timestamp plot
            # norm_latency_data = np.array(df_latency_i["latency_normalized"].tolist())
            # monitor_plot_instance(9, instance_id, 
            #     latency_mid_timestamp[sorted_indices], norm_latency_data[sorted_indices], "Normalized Latency at Mid-way time") 

            # 9. total TPS (prompt + gen)
            # Agg + Ending timestamp plot
            tps_interval = 5.0
            tps_timestamp_i, tps_data_i = aggregate_by_interval(
                df_latency_i["record_time"], df_latency_i["prompt_tokens"] + df_latency_i["generated_tokens"],
                tps_interval, method='sum')
            monitor_plot_instance(5, instance_id, tps_timestamp_i, tps_data_i, f"TPS Imd.")
            monitor_plot_instance(5, MAX_NUM_INSTANCES, tps_timestamp_i, tps_data_i, f"TPS Imd. - {instance_id}")



        ### E.B. All instance total plot, idx = MAX_NUM_INSTANCES+1
        # 1. TTFT
        # Agg plot
        ttft_interval = 5.0
        ttft_agg_timestamp, ttft_agg_data = aggregate_by_interval(
            df_latency["ttft_timestamp"], df_latency["TTFT"], ttft_interval, method='mean')
        monitor_plot_instance(2, MAX_NUM_INSTANCES+1, ttft_agg_timestamp, ttft_agg_data, f"TTFT per {ttft_interval}s - All")

        # 2. Normalized Latency
        # Agg plot
        norm_latency_interval = 5.0 
        norm_latency_agg_timestamp, norm_latency_agg_data = aggregate_by_interval(
            df_latency["record_time"], df_latency["latency_normalized"], norm_latency_interval, method='mean')
        monitor_plot_instance(3, MAX_NUM_INSTANCES+1, norm_latency_agg_timestamp, norm_latency_agg_data, f"Norm. Latency per {norm_latency_interval}s - All")

        # 3. RPS
        rps_interval = 5.0
        rps_timestamp, rps_data = aggregate_by_interval(
            df_latency["record_time"], df_latency["record_time"], rps_interval, method='count')
        monitor_plot_instance(4, MAX_NUM_INSTANCES+1, rps_timestamp, rps_data, f"RPS Imd. - All")

        # 4. instance numbers (monitor log would be empty if instance is close)
        monitor_plot_instance(0, MAX_NUM_INSTANCES+1, 
                              instance_nums_data["time"] - timestamp_base, 
                              instance_nums_data["value"], 
                              "Instances number")
        avg_instance_nums = np.mean(instance_nums_data["value"])
        monitor_plot_instance(0, MAX_NUM_INSTANCES+1,
                              [0, timestamp_end], 
                              [avg_instance_nums, avg_instance_nums], 
                              "Avg. Instances number")
        
        # 4.+. with RPS
        if workload == "Azure_code_peak":  
            rps_per_inst = 11.9
        elif workload == "Azure_conv_peak":
            rps_per_inst = 10.8
        monitor_plot_instance(4, MAX_NUM_INSTANCES+1, 
                              instance_nums_data["time"] - timestamp_base, 
                              np.array(instance_nums_data["value"]) * rps_per_inst,
                              "Instances number")
        monitor_plot_instance(4, MAX_NUM_INSTANCES+1,
                              [0, timestamp_end], 
                              [avg_instance_nums * rps_per_inst, avg_instance_nums * rps_per_inst], 
                              "Avg. Instances number")
                
 
        # 5. Total TPS
        tps_interval = 5.0
        tps_timestamp, tps_data = aggregate_by_interval(
            df_latency["record_time"], df_latency["prompt_tokens"] + df_latency["generated_tokens"],
            tps_interval, method='sum')
        monitor_plot_instance(5, MAX_NUM_INSTANCES+1, tps_timestamp, tps_data, f"TPS Imd. - All")


        # # 6. Optimal instance numbers

        # # 6.1. RPS: based on the max RPS in every scaling interval
        # # X-axis Discretization
        # scaling_intervals = [100]  # Would Determine the scaling frequency
        # # Y-axis Discretization: plot it on RPS plot, so scale up by 'rps_per_instance'
        # rps_range = np.percentile(rps_data, 99) 
        # rps_per_instance = rps_range / MAX_NUM_INSTANCES
        # print(f"rps_per_instance = {rps_range} / {MAX_NUM_INSTANCES} = {rps_per_instance}")

        # for interval in scaling_intervals:
        #     scaling_timestamps = range(0, int(timestamp_end), interval)
        #     max_rps_per_interval = [
        #         np.percentile( rps_data[ (rps_timestamp >= t) & (rps_timestamp < t + interval)], 90)
        #         for t in scaling_timestamps
        #     ]
        #     # max_discr_rps_per_interval = [
        #     #     math.ceil(rps / rps_per_instance) * rps_per_instance 
        #     #     for rps in max_rps_per_interval
        #     # ]
        #     max_discr_rps_per_interval = np.ceil(max_rps_per_interval / rps_per_instance) * rps_per_instance
        #     max_discr_rps_per_interval = [min(rps_range, rps) for rps in max_discr_rps_per_interval]

        #     axs[MAX_NUM_INSTANCES+1][4].step(
        #         scaling_timestamps, 
        #         max_discr_rps_per_interval,
        #         label=f"Optimal #Inst - RPS {interval}s",
        #         where='post')



        # # E.*.b+. Add max GPU space, in comparison to `current_token_usage`

        for i in range(MAX_NUM_INSTANCES+2):
            monitor_plot_instance(1, i, [0, timestamp_end], [GPU_max_tokens, GPU_max_tokens], "GPU_max_tokens")


        # E.*.b+. Unify y-axis range for TTFT, TBT
        def monitor_set_ylim(axs_id, ylim_min, ylim_max):
            for i in range(MAX_NUM_INSTANCES+2):
                if axs_id < num_subplots_0:
                    axs_0[i][axs_id].set_ylim(ylim_min, ylim_max)
                else:
                    axs_1[i][axs_id - num_subplots_0].set_ylim(ylim_min, ylim_max)
                axs[i][axs_id].set_ylim(ylim_min, ylim_max)

        # monitor_set_ylim(2, 0, 2 * pd_ratio_mean / num_instances + 4 * pd_ratio_std / num_instances)
        # monitor_set_ylim(3, 0, 2 * tbpt_mean + 4 * tbpt_std)
        # monitor_set_ylim(4, 0, 2 * ttft_mean + 4 * ttft_std)
        # monitor_set_ylim(5, 0, 2 * normalized_ttft_mean + 4 * normalized_ttft_std)
        # monitor_set_ylim(6, 0, 2 * tps_mean + 4 * tps_std)
        # monitor_set_ylim(7, 0, 2 * rps_mean + 4 * rps_std)


        # E.*.b+. x-axis all range
        for i in range(MAX_NUM_INSTANCES+2):
            for j in range(num_subplots):
                if j < num_subplots_0:
                    axs_0[i][j].set_xlim(0, timestamp_end)
                else:
                    axs_1[i][j - num_subplots_0].set_xlim(0, timestamp_end)
                axs[i][j].set_xlim(0, timestamp_end)


        ### Z. config and save
        for ax in axs_0.flat:
            if ax.lines: ax.legend(prop = {'size':6})
            ax.grid(True, which='both', axis='both')
        for ax in axs_1.flat:
            if ax.lines: ax.legend(prop = {'size':6})
            ax.grid(True, which='both', axis='both')
        for ax in axs.flat:
            if ax.lines: ax.legend(prop = {'size':6})
            ax.grid(True, which='both', axis='both')
        fig_0.tight_layout()
        fig_0.suptitle(f"Instance plot P0 - case_{case_id} - {result_timestamp}")
        fig_0.savefig(instance_fig_path.replace(".png", "_P0.png"), bbox_inches="tight", dpi=300)
        plt.close(fig_0)
        fig_1.tight_layout()
        fig_1.suptitle(f"Instance plot P1 - case_{case_id} - {result_timestamp}")
        fig_1.savefig(instance_fig_path.replace(".png", "_P1.png"), bbox_inches="tight", dpi=300)
        plt.close(fig_1)
        fig.tight_layout()
        fig.suptitle(f"Instance plot - case_{case_id} - {result_timestamp}")
        fig.savefig(instance_fig_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
    
    
    return 1


if __name__ == "__main__":
    for i in range(300, 350):
        exist = process_case(i)
        # if not exist:
        #     break
        print(i)
