import re
import datetime
import os
import numpy as np

def extract_scaler_info(case_dir):
    loads = []
    
    log_path = os.path.join(case_dir, "run.log")
    with open(log_path, "r") as f:
        log_lines = f.readlines()
        print(f"Log file {log_path} has {len(log_lines)} lines")

        # [INFO] [2025-03-08 15:22:36,429]  [scaler.py] : Scaler monitored average load: 883.25
        pattern = re.compile(
            r'\[INFO\] \[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}\]  \[\w+\.py\] : Scaler monitored average load: (\d+)'
        )
        for line in log_lines:
            match = pattern.search(line)
            if match:
                load = float(match.group(1))
                loads.append(load)

        if len(loads) == 0:
            print(f"No scaler info found in {log_path}")
        return loads



def process_case_load(case_id):
    ## 1. Extract
    case_dir = f"./cases/case_{case_id}"
    if not os.path.exists(case_dir):
        print(f"Case directory not found: {case_dir}")
        return 0

    loads = extract_scaler_info(case_dir)


    # ## 2. Optimal inst number
    # # X-axis Discretization
    # scaler_log_interval = 5
    # relative_timestamp_end = int(3600 / scaler_log_interval)
    # relative_interval = int(100 / scaler_log_interval)
    # # Y-axis Discretization
    # MAX_NUM_INSTANCES = 8
    # load_range = np.percentile(loads, 99) * 1.01
    # load_per_instance = load_range / MAX_NUM_INSTANCES
    # print(f"load_per_instance = {load_range} / {MAX_NUM_INSTANCES} = {load_per_instance}")

    # relative_timestamps = range(0, relative_timestamp_end, relative_interval)
    # max_loads_per_interval = [
    #     np.percentile(loads[i : i+relative_interval], 85) 
    #     for i in relative_timestamps
    # ]
    # discr_max_loads_per_interval = np.ceil(max_loads_per_interval / load_per_instance) * load_per_instance
    # discr_max_loads_per_interval = [
    #     min(load_range, w) for w in discr_max_loads_per_interval
    # ]


    ## 3. Ploting 
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(loads, label="loads")
    # plt.step(relative_timestamps, discr_max_loads_per_interval, 
    #     label="Optimal Num Instances", where='post')
    plt.legend()

    plt.savefig(f"{case_dir}/scaler_info.png")





if __name__ == '__main__':
    # process_case_wkpd(223)
    # process_case_load(215)
    process_case_load(214)