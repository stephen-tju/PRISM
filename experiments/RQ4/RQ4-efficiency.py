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



def process_case(case_id):
    case_dir = f"./case_{case_id}"
    if not os.path.exists(case_dir):
        return 0
    print("Processing case", case_id)

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
    df_latency = df.dropna(subset=['TTFT', 'latency', 'record_time']).copy()
    # status=200 / expection / dummyReq

    df_latency["frontend_overhead"] = df_latency["frontend_latency"] - df_latency["latency"]

    df_latency.loc[:, "norm_latency"] = df_latency["latency"] / df_latency["generated_tokens"]

    metrics_names = ["TTFT", "norm_latency", "latency", "frontend_overhead", "req_predictor_time"]
    metrics_dict = {}
    for metric_name in metrics_names:
        calculate_metrics(list(df_latency[metric_name]), metric_name, metrics_dict)
    # results = []
    # for metric_name, metric_json in metrics_dict.items():
    #     results.append((metric_json["mean"], metric_json["p99"]))
    # print(results)
    return metrics_dict


# def plot_results(metrics_dict):
#     plt.rcParams.update({'font.size': 25, "font.family": 'Times New Roman'})
#     scale = 2.4
#     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(scale * 4, 4), dpi=300)
#     metrics_names = ["TTFT", "norm_latency", "req_predictor_time", "frontend_overhead", ]
#     colors = ['#C2C1E0', '#F8DD85', '#AFCFA5', '#80A6E2']
#     max_value = 0
#     # for metric_name in metrics_names:
    
#     TTFT_value = metrics_dict['TTFT']['mean'] * 1000
#     max_value = max(max_value, TTFT_value)
#     print("TTFT: ", TTFT_value)
#     ax.barh(2, TTFT_value, color=colors[0], height=0.72)
#     ax.text(TTFT_value + 2, 2 * 1, "{:.1f}".format(TTFT_value), va='center', color='black', fontsize=20)


#     token_value = metrics_dict['norm_latency']['mean'] * 1000
#     total_value = metrics_dict['latency']['mean'] * 1000
#     # max_value = max(max_value, mean_value)
#     y_bottom = 1 - 0.71 / 2
#     y_top = 1 + 0.71 / 2
#     part_num = int((TTFT_value / token_value) + 1)
#     for part_id in range(part_num):
#         x_separator = part_id * token_value
#         ax.barh(1, token_value, left=x_separator, color=colors[1], height=0.72)
#         ax.plot([x_separator, x_separator], [y_bottom, y_top], color='black', linestyle='-', linewidth=1)

#     ax.barh(1, total_value - part_num*token_value, left=part_num*token_value, color=colors[1], height=0.72)

#     x1 = metrics_dict[metrics_names[2]]["mean"] * 1000
#     total = metrics_dict[metrics_names[3]]["mean"] * 1000
#     x2 = total - x1
#     print("Prediction: ", x1, "Other Overhead: ", x2)
#     ax.barh(3, x1, color=colors[2], height=0.8, label='Request Load\nPrediction')
#     ax.barh(3, x2, left=x1, color=colors[3], height=0.8, label='Other Overhead')
#     ax.text(total+2, 3, "{:.1f} + {:.1f} = {:.1f}".format(x1, x2, total), va='center', color='black', fontsize=20)
    

#     groups_name = ["Latency", "TTFT", "Schduling\nOverhead"]
#     # ax.set_xlim(0, max_value + 30)
#     # ax.set_xticks(np.arange(0, max_value + 30, 50))
#     ax.set_xlabel("Average time (ms)")
#     ax.set_yticks([1, 2, 3])
#     ax.set_yticklabels(groups_name)    
        
#     ax.legend(fontsize=22, handlelength=1)
#     plt.savefig("RQ3-efficiency.pdf", bbox_inches='tight')
#     plt.close()
        
def plot_results(metrics_dict):
    plt.rcParams.update({'font.size': 25, "font.family": 'Times New Roman'})
    scale = 2
    fig, (ax_left, ax_right) = plt.subplots(ncols=2, sharey=True, 
                                              figsize=(scale * 8, 4), dpi=300,
                                              gridspec_kw={'width_ratios': [2, 1], 'wspace': 0.1})
    # fig, (ax_left, ax_right) = plt.subplots(ncols=2, sharey=True, figsize=(scale * 8, 4), dpi=300)

    metrics_names = ["TTFT", "norm_latency", "req_predictor_time", "frontend_overhead"]
    colors = ['#C2C1E0', '#F8DD85', '#AFCFA5', '#80A6E2']

    TTFT_value = metrics_dict['TTFT']['mean']
    token_value = metrics_dict['norm_latency']['mean']
    total_value = metrics_dict['latency']['mean']
    print(total_value)

    print("TTFT: ", TTFT_value)

    part_num = int((TTFT_value / token_value) + 1)
    x_break = part_num * token_value  

    ax_left.barh(2, TTFT_value, color=colors[0], height=0.72)
    ax_left.text(TTFT_value + 0.005, 2, "{:.1f}ms".format(TTFT_value*1000), va='center', color='black', fontsize=20)

    y_bottom = 1 - 0.71 / 2
    y_top = 1 + 0.71 / 2
    for part_id in range(part_num):
        x_sep = part_id * token_value
        ax_left.barh(1, token_value, left=x_sep, color=colors[1], height=0.72)
        ax_left.plot([x_sep, x_sep], [y_bottom, y_top], color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    ax_left.text(token_value/2 - 0.018, 1, "{:.1f}ms".format(token_value*1000), va='center', color='black', fontsize=20)

    x_sep = total_value - token_value
    ax_right.plot([x_sep, x_sep], [y_bottom, y_top], color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax_right.barh(1, total_value, left=0, color=colors[1], height=0.72)
    ax_right.text(total_value + 0.005, 1, "{:.1f}s".format(total_value), va='center', color='black', fontsize=20)
    
    x1 = metrics_dict[metrics_names[2]]["mean"]
    total_sched = metrics_dict[metrics_names[3]]["mean"]
    x2 = total_sched - x1
    print("Prediction: ", x1, "Other Overhead: ", x2)
    ax_left.barh(3, x1, color=colors[2], height=0.8, label='Request Load\nPrediction')
    ax_left.barh(3, x2, left=x1, color=colors[3], height=0.8, label='Other Overhead')
    ax_left.text(total_sched + 0.005, 3, "{:.1f} + {:.1f} = {:.1f}ms".format(x1*1000, x2*1000, total_sched*1000), va='center', color='black', fontsize=20)
    
    groups_name = ["(Norm.)\nLatency", "TTFT", "Schdule\nOverhead"]
    ax_left.set_yticks([1, 2, 3])
    ax_left.set_yticklabels(groups_name)
    # ax_right.set_yticks([1, 2, 3])
    # ax_right.set_yticklabels([])

    fig.text(0.5, -0.05, "Average time (s)", ha='center', fontsize=30)

    
    ax_left.set_xlim(0, x_break * 1.05)
    ax_right.set_xlim(total_value - 0.11, total_value + 0.05)
    print(total_value - 0.1)
    start = np.ceil((total_value - 0.11) * 10) / 10
    ticks = np.arange(start, total_value + 0.05, 0.1)
    ax_right.set_xticks(ticks)
    ax_right.set_xticklabels([f'{tick:.1f}' for tick in ticks])

    
    ax_left.spines['right'].set_visible(False)
    ax_right.spines['left'].set_visible(False)
    
    d = .01  
    kwargs = dict(transform=ax_left.transAxes, color='k', clip_on=False)
    ax_left.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax_left.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=ax_right.transAxes)
    ax_right.plot((-d, d), (-d, +d), **kwargs)
    ax_right.plot((-d, d), (1 - d, 1 + d), **kwargs)
    
    plt.setp(ax_right.get_yticklines(), visible=False)
    
    handles, labels = ax_left.get_legend_handles_labels()
    ax_right.legend(handles, labels, fontsize=22, handlelength=1)
    
    plt.savefig("RQ4-efficiency.pdf", bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    metrics_dict = process_case(317)
    plot_results(metrics_dict)
    