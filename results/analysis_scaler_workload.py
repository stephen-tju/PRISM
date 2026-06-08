import re
import datetime
import os
import numpy as np

def extract_scaler_info(case_dir):
    
    log_path = os.path.join(case_dir, "run.log")
    with open(log_path, "r") as f:
        log_lines = f.readlines()
        print(f"Log file {log_path} has {len(log_lines)} lines")

        # [INFO] [2025-03-09 13:04:23,371]  [scaler.py] : Scaler prev groundtruth: 125734, next prediction: 135568
        pattern = re.compile(
            # r'Scaler prev groundtruth: (\d+), next prediction: (\d+)'
            r'\[INFO\] \[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}\]  \[\w+\.py\] : Scaler prev groundtruth: (\d+), next prediction: (\d+)'
        )
        groundtruths = []
        next_preds = []
        for line in log_lines:
            match = pattern.search(line)
            if match:
                groundtruth = int(match.group(1))
                next_pred = int(match.group(2))
                # print(f"{groundtruth}, {next_pred}")

                groundtruths.append(groundtruth)
                next_preds.append(next_pred)

        if len(groundtruths) == 0:
            print(f"No scaler info found in {log_path}")
    
    
        # # [INFO] [2025-03-10 05:00:06,768]  [scaler.py] : Scaled to 8 instances.
        # pattern = re.compile(
        #     r'\[INFO\] \[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}\]  \[\w+\.py\] : Scaled to (\d+) instances.'
        # )
        # actual_num_instances = []
        # for line in log_lines:
        #     match = pattern.search(line)
        #     if match:
        #         actual_num_inst = int(match.group(1)) 
        #         actual_num_instances.append(actual_num_inst)

        return groundtruths, next_preds


def process_case_wkpd(case_id):
    ## 1. Extract
    case_dir = f"./cases/case_{case_id}"
    if not os.path.exists(case_dir):
        print(f"Case directory not found: {case_dir}")
        return 0

    gt, npred = extract_scaler_info(case_dir)
    # print("Groundtruths:", gt)
    # print("Next Predictions:", npred)


    ## 2. Optimal inst number
    # X-axis Discretization
    relative_timestamp_end = int(3600 / 20)
    relative_interval = int(100 / 20)
    # Y-axis Discretization
    MAX_NUM_INSTANCES = 8
    workload_range = np.percentile(npred, 100) * 1.00
    workload_per_instance = workload_range / MAX_NUM_INSTANCES
    print(f"workload_per_instance = {workload_range} / {MAX_NUM_INSTANCES} = {workload_per_instance}")

    relative_timestamps = range(0, relative_timestamp_end, relative_interval)
    max_workloads_per_interval = [
        np.percentile(npred[i : i+relative_interval], 85) 
        for i in relative_timestamps
    ]
    discr_max_workloads_per_interval = np.ceil(max_workloads_per_interval / workload_per_instance) * workload_per_instance
    discr_max_workloads_per_interval = [
        min(workload_range, w) for w in discr_max_workloads_per_interval
    ]


    ## 3. Ploting 
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(gt, label="Groundtruth")
    plt.plot(npred, label="Next Prediction")
    plt.step(relative_timestamps, discr_max_workloads_per_interval, 
        label="Optimal Num Instances", where='post')
    plt.legend()

    plt.savefig(f"{case_dir}/scaler_info_rel{relative_interval}_more.png")

if __name__ == '__main__':
    process_case_wkpd(236)
    process_case_wkpd(237)
