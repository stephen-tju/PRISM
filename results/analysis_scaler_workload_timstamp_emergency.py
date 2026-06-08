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
        pattern_scaler = re.compile(
            r'\[INFO\] \[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\]  \[\w+\.py\] : Scaler prev groundtruth: (\d+), next prediction: (\d+)'
        )
        # # [INFO] [2025-03-10 05:00:06,768]  [scaler.py] : Scaled to 8 instances.
        pattern_instances = re.compile(
            r'\[INFO\] \[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\]  \[\w+\.py\] : .+caled to (\d+) instances.'
        )
    
        timestamps = []
        groundtruths = []
        next_preds = []
        instance_timestamps = []
        actual_num_instances = []
        
        for line in log_lines:
            match_scaler = pattern_scaler.search(line)
            if match_scaler:
                timestamps.append(datetime.datetime.strptime(match_scaler.group(1), "%Y-%m-%d %H:%M:%S,%f"))
                groundtruths.append(int(match_scaler.group(2)))
                next_preds.append(int(match_scaler.group(3)))
            
            match_instance = pattern_instances.search(line)
            if match_instance:
                instance_timestamps.append(datetime.datetime.strptime(match_instance.group(1), "%Y-%m-%d %H:%M:%S,%f"))
                actual_num_instances.append(int(match_instance.group(2)))
        
        if not groundtruths:
            print(f"No scaler info found in {log_path}")
        
        return timestamps, groundtruths, next_preds, instance_timestamps, actual_num_instances


def process_case_wkpd(case_id):
    ## 1. Extract
    case_dir = f"./cases/case_{case_id}"
    if not os.path.exists(case_dir):
        print(f"Case directory not found: {case_dir}")
        return 0

    timestamps, gt, npred, instance_timestamps, actual_num_instances = extract_scaler_info(case_dir)
    relative_timestamps = [(t - timestamps[0]).total_seconds() for t in timestamps]
    instance_relative_timestamps = [(t - timestamps[0]).total_seconds() for t in instance_timestamps]
    print(f"Scaler workload #gt {len(gt)}, #npred {len(npred)}, #ts {len(relative_timestamps)}")
    print(f"Scaler actual #inst {len(actual_num_instances)}, #ts {len(instance_relative_timestamps)}")

    # ## 2. Optimal inst number
    # # X-axis Discretization
    # relative_timestamp_end = int(3600 / 20)
    # relative_interval = int(100 / 20)
    # Y-axis Discretization
    MAX_NUM_INSTANCES = 8
    workload_range = np.percentile(npred, 100) * 1.00
    workload_per_instance = workload_range / MAX_NUM_INSTANCES
    print(f"workload_per_instance = {workload_range} / {MAX_NUM_INSTANCES} = {workload_per_instance}")

    MIU_A = 35000 * 2
    workload_range = MAX_NUM_INSTANCES * MIU_A
    workload_per_instance = MIU_A
    print(f"workload_per_instance = {workload_range} / {MAX_NUM_INSTANCES} = {workload_per_instance}")

    # relative_opt_timestamps = range(0, relative_timestamp_end, relative_interval)
    # max_workloads_per_interval = [
    #     np.percentile(npred[i : i+relative_interval], 85) 
    #     for i in relative_timestamps
    # ]
    # discr_max_workloads_per_interval = np.ceil(max_workloads_per_interval / workload_per_instance) * workload_per_instance
    # discr_max_workloads_per_interval = np.clip(discr_max_workloads_per_interval, 1, workload_range)


    ## 3. Ploting 
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(relative_timestamps, gt, label="Groundtruth")
    plt.plot(relative_timestamps, npred, label="Next Prediction")
    # plt.step(relative_opt_timestamps, discr_max_workloads_per_interval, 
    #     label="Optimal Num Instances", where='post')
    plt.step(instance_relative_timestamps, np.array(actual_num_instances) * workload_per_instance, 
        label="Actual Num Instances", where='post', linestyle="--")
    plt.legend()

    plt.savefig(f"{case_dir}/scaler_info.png")

if __name__ == '__main__':
    # process_case_wkpd(277)
    # process_case_wkpd(276)
    process_case_wkpd(343)

