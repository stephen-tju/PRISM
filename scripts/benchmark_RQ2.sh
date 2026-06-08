#!/bin/bash 


model_name="meta-llama/Llama-2-7b-hf"

# loads=("ShareGPT")
# loads=("Azure_code")
# loads=("Azure_conv")
loads=("workload_trace")

workloads=()
workload_sampling_intervals=()
workloads+=("Azure_code")
workload_sampling_intervals+=(27)
workloads+=("Azure_conv")
workload_sampling_intervals+=(120)

L="0"
R="1"
# L="0.5"
# R="0.8"
# L="0.2"
# R="0.5"

# request_nums=(3000)
# request_nums=(2000)
# request_nums=(100000)
request_nums=(80000)

benchmark_durations=(7200)
# benchmark_durations=(3600)
# benchmark_durations=(1000)
# benchmark_durations=(300)

qps_values=(1)

scheduler_policies=("preserve")
# scheduler_policies=("preserve" "least_requests" "least_utilization" "round_robin")
scheduler_params=(2)


scaler_policies=()
scaler_intervals=()
scaler_policies+=("proactive")
scaler_intervals+=(20)
scaler_policies+=("reactive") 
scaler_intervals+=(10)
scaler_policies+=("hybrid") 
scaler_intervals+=(10)
scaler_policies+=("pass")
scaler_policies+=(10)
scaler_policies+=("llumnix") 
scaler_intervals+=(10)
scaler_policies+=("preserve") 
scaler_intervals+=(5)
scaler_policies+=("preserve_err") 
scaler_intervals+=(5)

req_predictor_policies=("ground_truth")
# req_predictor_policy=("load_predictor")
# req_predictor_policy=("load_predictor" "ground_truth")

# initial number of instances
num_instances=(4)


cd ../LLMServe/
for request_num in "${request_nums[@]}"; do
  for benchmark_duration in "${benchmark_durations[@]}"; do
    for load in "${loads[@]}"; do
      for workload_i in "${!workloads[@]}"; do
        workload=${workloads[$workload_i]}
        workload_sampling_interval=${workload_sampling_intervals[$workload_i]}

        for num_instance in "${num_instances[@]}"; do
          for scheduler_policy in "${scheduler_policies[@]}"; do
            for scaler_i in "${!scaler_policies[@]}"; do
              scaler_policy=${scaler_policies[$scaler_i]}
              scaler_interval=${scaler_intervals[$scaler_i]}

              for qps in "${qps_values[@]}"; do
                for req_predictor_policy in "${req_predictor_policies[@]}"; do
                  for scheduler_param in "${scheduler_params[@]}"; do
                    echo "Benchmark (Scaling) on $load, $workload[$L, $R], $request_num reqs, $benchmark_duration secs, $num_instance inst, $qps qps, policy = $scheduler_policy($scheduler_param) x $req_predictor_policy x $scaler_policy($scaler_interval)"

                    python benchmark.py \
                      --instance_configurations_path "../instance_configurations_8.json" \
                      --model_name "$model_name" \
                      --result_dir "../results/cases/" \
                      --request_num "$request_num" \
                      --benchmark_duration "$benchmark_duration" \
                      --load "$load" \
                      --load_dataset_path "../data/workloads/$load/cleaned.csv" \
                      --workload "$workload" \
                      --workload_trace_path "../data/workloads/$workload/cleaned.csv" \
                      --workload_timescale 24 \
                      --workload_sampling_interval "$workload_sampling_interval" \
                      --workload_trace_range_L "$L" \
                      --workload_trace_range_R "$R" \
                      --qps "$qps" \
                      --num_instances "$num_instance" \
                      --scheduler_policy "$scheduler_policy" \
                      --scheduler_param "$scheduler_param" \
                      --scaler_policy "$scaler_policy" \
                      --scaler_interval "$scaler_interval" \
                      --req_predictor_policy "$req_predictor_policy" \
                      --max_model_len 4096 \
                      --max_num_seqs 128 \
                      --max_num_batched_tokens 8192 
                    sleep 30
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done
