#!/bin/bash 

model_name="meta-llama/Llama-2-7b-hf"

loads=("ShareGPT")
# loads=("Azure_code" "Azure_conv")
# loads=("Azure_conv")
# loads=("workload_trace")

workloads=("poisson")
# workloads=("Azure_code")
# workloads=("Azure_conv")

request_nums=(3000)

qps_values=(8.7 9 9.3 9.6 9.9 10.2 10.5)

scheduler_policies=("preserve" "least_requests" "sia" "magnus_mem_restrained" "least_utilization" "round_robin")
scheduler_params=(2)

scaler_policies=("none")

req_predictor_policies=("load_predictor")

num_instances=(4)



cd ../LLMServe/

for request_num in "${request_nums[@]}"; do
  for load in "${loads[@]}"; do
    for workload in "${workloads[@]}"; do
      for num_instance in "${num_instances[@]}"; do
        for scheduler_policy in "${scheduler_policies[@]}"; do
          for scaler_policy in "${scaler_policies[@]}"; do
            for qps in "${qps_values[@]}"; do
              for req_predictor_policy in "${req_predictor_policies[@]}"; do
                for scheduler_param in "${scheduler_params[@]}"; do
                  echo "Benchmark on $load, $workload, $request_num requests, $num_instance instances, $qps qps, policy = $scheduler_policy($scheduler_param) x $req_predictor_policy x $scaler_policy"

                  python benchmark.py \
                    --instance_configurations "../instance_configurations_4.json" \
                    --model_name "$model_name" \
                    --result_dir "../results/cases/" \
                    --load "$load" \
                    --load_dataset_path "../data/datasets/$load/cleaned.csv" \
                    --workload "$workload" \
                    --request_num "$request_num" \
                    --qps "$qps" \
                    --num_instances "$num_instance" \
                    --scheduler_policy "$scheduler_policy" \
                    --scheduler_param "$scheduler_param" \
                    --scaler_policy "$scaler_policy" \
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
