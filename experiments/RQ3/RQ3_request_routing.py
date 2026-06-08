import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

def read_data(model_name, qps_values, request_nums, num_instanceses, scheduler_policies):
    used_case_ids = set()
    data_records = []

    case_folders = [f for f in os.listdir(f"./{model_name}") if os.path.isdir(os.path.join(model_name, f)) and f.startswith("case_")]
    
    for case_folder in case_folders:
        case_id = int(case_folder.split('_')[1])
        case_folder_path = os.path.join(model_name, case_folder)
        
        metrics_path = os.path.join(case_folder_path, "metrics.json")
        if os.path.isfile(metrics_path):
            with open(metrics_path, "r") as file:
                metrics = json.load(file)
                main_info, instance_info, metrics_info = metrics
                assert len(metrics_info) > 0, "No metrics found"

                if (main_info['request_num'] in request_nums and
                    main_info['qps'] in qps_values and
                    instance_info['num_instances'] in num_instanceses and
                    instance_info['scheduler_policy'] in scheduler_policies):

                    metrics_info["Norm Latency"]["SLO maintenance"] = 100 - metrics_info["Norm Latency"]["slo_violation_perc"]
                    
                    record = {
                        "case_id": case_id,
                        "request_num": main_info["request_num"],
                        "qps": main_info["qps"],
                        "num_instances": instance_info["num_instances"],
                        "scheduler_policy": instance_info["scheduler_policy"],
                        "scheduler_param": instance_info["scheduler_param"],
                        "req_predictor_policy": instance_info["req_predictor_policy"],
                        "metrics_info": metrics_info,
                    }
                    data_records.append(record)
                    used_case_ids.add(case_id)

    df = pd.DataFrame(data_records)
    assert len(df) > 0, "No data found for the specified case ids"

    return df, used_case_ids

def process_data(df, curve_metrics_val):
    curve_metrics = ["scheduler_policy"]
    curve_list = []
    for metric_combination in product(*curve_metrics_val):
        curve_list.append(list(zip(curve_metrics, metric_combination)))
    
    df = df.sort_values(by='qps', ascending=True)

    return curve_list, df

def plot_metrics_data(df, curve_list, plot_metrics, plot_quantile, model_name, axs, row_idx, row_offset=0):
    colors = {
        "preserve": '#428B2C',
        "round_robin": '#F79B25',
        "least_requests": '#4e75b0',
        "least_utilization": '#C52A20',
    }
    markers = {
        "preserve": 's',
        "round_robin": 'x',
        "least_requests": '^',
        "least_utilization": 'o',
    }
    curve_label_map = {
        "preserve": "PreServe",
        "round_robin": "Round Robin",
        "least_requests": "Least Requests",
        "least_utilization": "Least Utilization",
    }
    
    for curve_idx, curve in enumerate(curve_list):
        df_curve = df.copy()
        for metric, value in curve:
            df_curve = df_curve[df_curve[metric] == value]
        df_curve.reset_index(drop=True, inplace=True)

        if len(df_curve) == 0:
            continue

        grouped_df = []
        for i in range(len(plot_metrics)): 
            metric, quantile = plot_metrics[i], plot_quantile[i]
            temp_df = df_curve[["qps", "metrics_info"]].copy()
            temp_df["metric_value"] = temp_df["metrics_info"].apply(lambda x: 
                x[metric][quantile] if metric in x and quantile in x[metric] else np.nan
            )
            temp_df.drop(columns=["metrics_info"], inplace=True)

            avg_df = temp_df.groupby("qps", as_index=False).agg({"metric_value": "mean"})
            grouped_df.append((i, avg_df))

        curve_label = ", ".join([f"{m[1]}" for m in curve])
        color = colors[curve_label]
        mark = markers[curve_label]
        for i, avg_df in grouped_df:
            ax = axs[row_offset + i]
            avg_df = avg_df.dropna().sort_values(by="qps")
            if len(avg_df) == 0:
                continue
            if i == 0:
                ax.plot(avg_df["qps"], avg_df["metric_value"], color=color, linewidth=1.5, linestyle='--', marker=mark, markersize=5, label=curve_label_map[curve_label])
            else:
                ax.plot(avg_df["qps"], avg_df["metric_value"], color=color, linewidth=1.5, linestyle='--', marker=mark, markersize=5)
            ax.tick_params(axis='both', which='major')

    titles = ["TTFT-P99", "TTFT-Mean", "Norm. Latency-P99", "Norm. Latency-Mean", "SLO maintenance"]
    if model_name == "LLaMA_2_7B":
        for i in range(5):
            ax = axs[row_offset + i]
            ax.grid(color='lightgray', axis="y", zorder=1, alpha=0.5)
            ax.grid(color='lightgray', axis="x", zorder=1, alpha=0.5)
            if i != 4:
                ax.set_ylabel("Latency (s)", fontsize=22)
            elif i == 4:
                ax.set_ylabel("Proportion (%)", fontsize=22)
            if row_idx == 0:
                ax.set_title(titles[i], fontsize=28)
                ax.set_xticks([7.5, 8.1, 8.7, 9.3, 9.9, 10.5])
                ax.set_xticklabels(["7.5", "8.1", "8.7", "9.3", "9.9", "10.5"], fontsize=22)
                # ax.set_xticklabels([])
            if row_idx == 1:
                ax.set_xlabel("QPS", fontsize=22)
                ax.set_xticks([7.5, 8.1, 8.7, 9.3, 9.9, 10.5])
                ax.set_xticklabels(["7.5", "8.1", "8.7", "9.3", "9.9", "10.5"], fontsize=22)
            if row_idx == 0 and i == 0:
                ax.text(-0.15, 0.5, 'LLaMA-2-7B', ha='center', va='center', rotation=90, transform=ax.transAxes, fontsize=30)
                # ax.text(0.5, -0.1, 'LLaMA-2-7B', ha='center', va='center', transform=ax.transAxes, fontsize=25)
            if row_idx == 1 and i == 0:
                ax.text(-0.15, 0.5, 'LLaMA-2-13B', ha='center', va='center', rotation=90, transform=ax.transAxes, fontsize=30)

        
        if row_idx == 0:
            for i in range(5):
                ax = axs[row_offset + i]
                if i == 0:
                    ax.set_ylim(0, 49)
                    ax.set_yticks([0, 12, 24, 36, 48])
                    ax.set_yticklabels(["0", "12", "24", "36", "48"], fontsize=22)
                elif i == 1:
                    ax.set_ylim(0, 12.3)
                    ax.set_yticks([0, 3, 6, 9, 12])
                    ax.set_yticklabels(["0", "3", "6", "9", "12"], fontsize=22)
                elif i == 2:
                    ax.set_ylim(0, 2.04)
                    ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
                    ax.set_yticklabels(["0", "0.5", "1.0", "1.5", "2.0"], fontsize=22)
                elif i == 3:
                    ax.set_ylim(0.05, 0.254)
                    ax.set_yticks([0.05, 0.10, 0.15, 0.20, 0.25])
                    ax.set_yticklabels(["0.05", "0.10", "0.15", "0.20", "0.25"], fontsize=22)
                elif i == 4:
                    ax.set_ylim(84, 101)
                    ax.set_yticks([85, 90, 95, 100])
                    ax.set_yticklabels(["85", "90", "95", "100"], fontsize=22)        

    elif model_name == "LLaMA_2_13B":
        for i in range(5):
            ax = axs[row_offset + i]
            ax.grid(color='lightgray', axis="y", zorder=1, alpha=0.5)
            ax.grid(color='lightgray', axis="x", zorder=1, alpha=0.5)
            if i != 4:
                ax.set_ylabel("Latency (s)", fontsize=22)
            elif i == 4:
                ax.set_ylabel("Proportion (%)", fontsize=22)
            if row_idx == 0:
                ax.set_title(titles[i], fontsize=28)
                ax.set_xticks([9.0, 10.0, 11.0, 12.0, 13.0, 14.0])
                ax.set_xticklabels(["9.0", "10.0", "11.0", "12.0", "13.0", "14.0"], fontsize=22)
                # ax.set_xticklabels([])
            if row_idx == 1:
                ax.set_xlabel("QPS", fontsize=22)
                ax.set_xticks([9.0, 10.0, 11.0, 12.0, 13.0, 14.0])
                ax.set_xticklabels(["9.0", "10.0", "11.0", "12.0", "13.0", "14.0"], fontsize=22)
            if row_idx == 0 and i == 0:
                ax.text(-0.15, 0.5, 'LLaMA-2-7B', ha='center', va='center', rotation=90, transform=ax.transAxes, fontsize=30)
                # ax.text(0.5, -0.1, 'LLaMA-2-7B', ha='center', va='center', transform=ax.transAxes, fontsize=25)
            if row_idx == 1 and i == 0:
                ax.text(-0.15, 0.5, 'LLaMA-2-13B', ha='center', va='center', rotation=90, transform=ax.transAxes, fontsize=30)

        if row_idx == 1:
            for i in range(5):
                ax = axs[row_offset + i]
                if i == 0:
                    ax.set_ylim(0, 25.2)
                    ax.set_yticks([0, 6, 12, 18, 24])
                    ax.set_yticklabels(["0", "6", "12", "18", "24"], fontsize=22)
                elif i == 1:
                    ax.set_ylim(0, 3.65)
                    ax.set_yticks([0, 0.9, 1.8, 2.7, 3.6])
                    ax.set_yticklabels(["0", "0.9", "1.8", "2.7", "3.6"], fontsize=22)
                elif i == 2:
                    ax.set_ylim(0, 1.03)
                    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
                    ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1.0"], fontsize=22)
                elif i == 3:
                    ax.set_ylim(0.05, 0.155)
                    ax.set_yticks([0.05, 0.07, 0.09, 0.11, 0.13, 0.15])
                    ax.set_yticklabels(["0.05", "0.07", "0.09", "0.11", "0.13", "0.15"], fontsize=22)
                elif i == 4:
                    ax.set_ylim(90.5, 100.4)
                    ax.set_yticks([91, 94, 97, 100])
                    ax.set_yticklabels(["91", "94", "97", "100"], fontsize=22)             

        # ax.tick_params(axis='both', which='major')
        
        

    handles, labels = axs[row_offset].get_legend_handles_labels()
    desired_order = ["Round Robin", "Least Requests", "Least Utilization", "PreServe"]

    label_to_handle = dict(zip(labels, handles))
    print(labels)
    new_handles = [label_to_handle[label] for label in desired_order]
    new_labels = desired_order

    return new_handles, new_labels

def process_and_plot_metrics(model_name1, model_name2, qps_values1, qps_values2, request_nums, num_instanceses, scheduler_policies, plot_metrics, plot_quantile):
    df1, _ = read_data(model_name1, qps_values1, request_nums, num_instanceses, scheduler_policies)
    print(f"Model 1: {model_name1}, Number of cases: {len(df1)}")

    curve_list1, df1 = process_data(df1, [scheduler_policies])

    plt.rcParams.update({'font.size': 20, "font.family": 'Times New Roman'})
    scale = 2.4

    if model_name2 is None:
        row, col = 1, 5
        fig, axs = plt.subplots(nrows=row, ncols=col, figsize=(scale * 4 * col + 0.25 * (col-1), 4 * row), dpi=300)
        fig.subplots_adjust(hspace=0.15, wspace=0.15)

        handles1, labels1 = plot_metrics_data(df1, curve_list1, plot_metrics, plot_quantile, model_name1, axs, row_idx=0)
    else:
        df2, _ = read_data(model_name2, qps_values2, request_nums, num_instanceses, scheduler_policies)
        print(f"Model 2: {model_name2}, Number of cases: {len(df2)}")

        curve_list2, df2 = process_data(df2, [scheduler_policies])

        row, col = 2, 5
        fig, axs = plt.subplots(nrows=row, ncols=col, figsize=(scale * 4 * col + 0.25 * (col-1), 4 * row), dpi=300)
        fig.subplots_adjust(hspace=0.15, wspace=0.15)

        handles1, labels1 = plot_metrics_data(df1, curve_list1, plot_metrics, plot_quantile, model_name1, axs[0], row_idx=0)
        handles2, labels2 = plot_metrics_data(df2, curve_list2, plot_metrics, plot_quantile, model_name2, axs[1], row_idx=1)

    fig.legend(handles1, labels1, loc='upper center', prop={'size': 25}, ncol=5, bbox_to_anchor=(0.5, 1.05))
    plt.savefig(f"./RQ3_request_routing.pdf", bbox_inches='tight')
    plt.savefig(f"./RQ3_request_routing.png", bbox_inches='tight')
    plt.close()

process_and_plot_metrics(
    model_name1="LLaMA_2_7B",
    model_name2=None,
    qps_values1=[7.5, 7.8, 8.1, 8.4, 8.7, 9, 9.3, 9.6, 9.9, 10.2, 10.5, 10.8],
    qps_values2=None,
    request_nums={2000, 3000},
    num_instanceses={4},
    scheduler_policies={"preserve", "round_robin", "least_requests", "least_utilization"},
    plot_metrics=["TTFT", "TTFT", "Norm Latency", "Norm Latency", "Norm Latency"],
    plot_quantile=["p99", "mean", "p99", "mean", "SLO maintenance"]
)
