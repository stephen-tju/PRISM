import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

compare_name = "avg_7B"

# Directly traverse the case to see if it meets the wrange, only filter, no defined order
case_ids = set(list(range(0, 500))) 
qps_values_list = list(np.round(np.arange(9, 15, 0.5), 1))
qps_values = set(qps_values_list)
print(f"target qps values: {qps_values}")
# request_nums = {3000}
request_nums = {2000}
num_instanceses = {4}
# loads = {"Azure_code", "Azure_conv"}
# loads = {"Azure_code"}
loads = {"ShareGPT"}
workloads = {"poisson"}
# scheduler_policies = {"load_5", "round_robin", "least_loaded", "min_utilization"}
scheduler_policies = ["preserve", "round_robin", "least_requests", "least_utilization"]
# scheduler_policies = ["preserve", "least_requests"]
# scheduler_policies = {"preserve"}
scheduler_params = {2.0}
req_predictor_policies = {"load_predictor"}

# Different curves come from which metrics? Free combination, each combination will be a curve
curve_metrics, curve_metrics_val = [], []
curve_metrics.append("scheduler_policy")
curve_metrics_val.append(scheduler_policies)
# curve_metrics.append("scheduler_param")
# curve_metrics_val.append(scheduler_params)
# curve_metrics.append("req_predictor_policy")
# curve_metrics_val.append(req_predictor_policies)
# curve_metrics.append("load")
# curve_metrics_val.append(loads)
# curve_metrics.append("request_num")
# curve_metrics_val.append(request_nums)



curve_list = []
for metric_combination in product(*curve_metrics_val):
    curve_list.append(list(zip(curve_metrics, metric_combination)))
print(f"Possible curves:")
for curve in curve_list: 
    print(curve)

data_records = []

for case_id in case_ids:
    case_folder = f"./cases/case_{case_id}"
    if os.path.isdir(case_folder):
        metrics_path = os.path.join(case_folder, "metrics.json")
        if os.path.isfile(metrics_path):
            with open(metrics_path, "r") as file:
                metrics = json.load(file)
                main_info, instance_info, metrics_info = metrics
                assert len(metrics_info) > 0, "No metrics found"

                # filtering
                if (main_info['request_num'] in request_nums and
                    main_info['qps'] in qps_values and
                    main_info['load'] in loads and
                    main_info['workload'] in workloads and
                    instance_info['num_instances'] in num_instanceses and
                    instance_info['scheduler_policy'] in scheduler_policies and
                    instance_info['scheduler_param'] in scheduler_params and 
                    instance_info['req_predictor_policy'] in req_predictor_policies):
                    
                    record = {
                        "case_id": case_id,
                        "request_num": main_info["request_num"],
                        "qps": main_info["qps"],
                        "load": main_info["load"],
                        "workload": main_info["workload"],
                        "num_instances": instance_info["num_instances"],
                        "scheduler_policy": instance_info["scheduler_policy"],
                        "scheduler_param": instance_info["scheduler_param"],
                        "req_predictor_policy": instance_info["req_predictor_policy"],
                        "metrics_info": metrics_info,
                    }
                    data_records.append(record)

df = pd.DataFrame(data_records)
assert len(df) > 0, "No data found for the specified case ids"

df = df.sort_values(by='qps', ascending=True)
print(f"#Records = {len(df)}")

# slo_maintenance = 100 - slo_violation_perc 


# Drawing
plot_metrics = ["TTFT", "TBT", "Latency", "Norm Latency", "RPS"]
plot_quantile = ["p99", "p95", "p90", "slo_violation_perc", "mean"]
w, h = len(plot_metrics), len(plot_quantile)

fig, axs = plt.subplots(w, h, figsize=(16, 14), dpi=400)

custom_ylim = {
#     # TTFT
    # (0, 0): (0, 15),
    # (0, 1): (0, 6),
# #     (0, 2): (0, 0.8),
#     (0, 4): (0, 2),
#     # NormLatency
#     (3, 0): (0, 0.5),
#     (3, 1): (0, 0.3),
}

# colors = plt.cm.get_cmap("tab10", len(curve_list))
colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]


for curve_idx, curve in enumerate(curve_list):
    df_curve = df.copy()
    for metric, value in curve:
        df_curve = df_curve[df_curve[metric] == value]
    df_curve.reset_index(drop=True, inplace=True)

    if len(df_curve) == 0:
        continue

    print(f"Drawing curve: {curve}")
    curve_label = ", ".join([f"{m[1]}" for m in curve])
    # color = colors(curve_idx)
    color = colors[curve_idx]

    grouped_df = []
    for i, metric in enumerate(plot_metrics):
        for j, quantile in enumerate(plot_quantile):
            temp_df = df_curve[["qps", "metrics_info"]].copy()
            temp_df["metric_value"] = temp_df["metrics_info"].apply(lambda x: x[metric][quantile] if metric in x and quantile in x[metric] else np.nan)
            temp_df.drop(columns=["metrics_info"], inplace=True)

            avg_df = temp_df.groupby("qps", as_index=False).agg({"metric_value": "mean"})
            grouped_df.append(((i, j), avg_df))

    for (i, j), avg_df in grouped_df:
        ax = axs[i, j]
        ylim = custom_ylim.get((i, j), None)

        avg_df = avg_df.dropna().sort_values(by="qps")
        if len(avg_df) == 0:
            continue

        ax.plot(avg_df["qps"], avg_df["metric_value"], color=color, linewidth=1.5, linestyle='-', label=curve_label)
        ax.set_title(f"{plot_metrics[i]}-{plot_quantile[j]}", fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=6)

axs[0,0].legend(fontsize=4)

# ylim
for i in range(w):
    for j in range(h):
        if (i, j) in custom_ylim:
            axs[i, j].set_ylim(custom_ylim[(i, j)])

# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.tight_layout()
plt.savefig(f"figures-compare/{compare_name}.png")
plt.close()
