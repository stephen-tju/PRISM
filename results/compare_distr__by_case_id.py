import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

case_ids_to_compare = {
    183: "2k-util",
    184: "2k-LL",
    185: "3k-util",
    187: "3k-LL",
    188: "4k-util",
    189: "4k-LL",
}
# ## QPS=16, ShareGPT(16,1024)
#     60: "GT_current_token_usage",
#     61: "GT_incoming_prefill_tokens"
#     62: "RR",
# }
# ## QPS=8, ShareGPT(16,2048)
#     52: "RR",
#     51: "GT_current_token_usage",
#     57: "GPU_cache_usage",
#     63: "GT_imminent_token_usage",
# }



def procecss_result_file(case_path, result_file):
    result_path = os.path.join(case_path, result_file)
    results = []
    try:
        with open(result_path, "r") as file:
            lines = file.readlines()
    except:
        print(f"Error: unable to read result file: {result_path}")
        return pd.DataFrame([])

    for line in lines:
        record = json.loads(line.strip())
        results.append(record)

    return pd.DataFrame(results)


def main():
    fig_dir = "./figures-compare"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    records = {
        'TTFT':[], 'TBPT':[], 'latency':[], 'norm_latency':[], 'itl_interval':[], 
        'time_after_last':[], 'RPS_precise':[], 'RPS_bucket':[]
    }
    for case_dirname in os.listdir("./cases/"):
        if int(case_dirname.split("_")[1]) in case_ids_to_compare:
            # Read result file
            case_path = os.path.join("./cases", case_dirname)
            files = os.listdir(case_path)
            result_files = [f for f in files if f.startswith("result") and f.endswith(".json")]
            if len(result_files) == 0:
                print(f"Error: case {case_path} does not have result file, add empty records.")
                for metric in records.keys():
                    records[metric].append([])
                continue
            
            df = procecss_result_file(case_path, result_files[-1])

            # preprocess
            df = df.dropna(subset=['TTFT', 'TBPT', 'latency', 'itl'])
            if len(df) == 0:
                print(f"Error: case {case_path} has no valid records, add empty records.")
                continue

            df.loc[:,'norm_latency'] = df['latency'] / df['generated_tokens']

            df.loc[:,'time_after_last'] = df['record_time'] - df['record_time'].shift(1)
            df.loc[0,'time_after_last'] = 0

            # record per request
            for metric in ['TTFT', 'TBPT', 'latency', 'norm_latency', 'time_after_last']:
                records[metric].append(df[metric].values)
            
            itl_intervals = np.concatenate(df['itl'].apply(
                lambda x: np.diff(x) if isinstance(x, list) and len(x) > 1 else []
            ))
            records['itl_interval'].append(itl_intervals)

            # RPS_precise: per request
            df.loc[:,'RPS_precise'] = 1 / df['time_after_last']
            records['RPS_precise'].append(df['RPS_precise'].values[1:])

            # RPS_bucket: record_time by n seconds
            n = 1
            min_record_time, max_record_time = df['record_time'].min(), df['record_time'].max()
            req_buckets = np.zeros(int((max_record_time - min_record_time) / n) + 1)
            for record_time in df['record_time']:
                req_buckets[int((record_time - min_record_time) / n)] += 1
            records['RPS_bucket'].append(req_buckets)


    # ploting CDF
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    for i, metric in enumerate(
        ['TTFT', 'norm_latency', 'RPS_bucket', 'RPS_precise', 'itl_interval', 'TBPT']
    ):
        plot_x, plot_y = i // 2, i % 2
        ax = axs[plot_x, plot_y]

        for i in range(len(records[metric])):
            data_list = records[metric][i]
            case_id = list(case_ids_to_compare.keys())[i]
            if len(data_list) == 0:
                continue
            hist, bin_edges = np.histogram(data_list, bins=50)
            cumsum = np.cumsum(hist)
            cdf_line = ax.plot(bin_edges[1:], cumsum / np.sum(hist) * 100, alpha=0.5, label=f"{case_ids_to_compare[case_id]}")
        
        ax.set_title(f"{metric} CDFs")
        ax.set_xlabel(metric)
        ax.set_ylabel("Frequency")
        ax.legend()
    

    # save figure
    # name_postfix = "_".join(case_ids_to_compare.keys())
    name_postfix = "_".join([case_ids_to_compare[case_id] for case_id in case_ids_to_compare.keys()])
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/cdf_case_{name_postfix}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()