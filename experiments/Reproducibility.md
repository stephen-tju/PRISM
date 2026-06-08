# Reproducibility

## Prerequisite

To reproduce any of the following experiment results, you should first download and preprocess the datasets.

```bash
cd ./data/

# Download the workload and load datasets
./run.sh

# Preprocess the load dataset: ShareGPT
python preprocess_datasets.py \
	--min_input_tokens 16 \
	--min_out_tokens 16 \
	--max_input_tokens 4096 \
	--max_out_tokens 4096 \
	--min_total_tokens 32 \
	--max_total_tokens 4096 \
	--tokenizer_name "meta-llama/Llama-2-7b-hf"

# Preprocess the load dataset: Azure_code & Azure_conv
python preprocess_workloads.py
```

To use the benchmark results we provided, you should first download and unzip them from Google Drive. 

```bash
# https://drive.google.com/file/d/1iL-ZOUxX_D2uYvVHIGVOAZxUr8J2IsuH/view?usp=sharing
fid=1iL-ZOUxX_D2uYvVHIGVOAZxUr8J2IsuH
python download_gdrive.py $fid ./PreServe_benchmark_results.zip
unzip PreServe_benchmark_results.zip
```

To run the `benchmark.py` to generate your local benchmark results, you should first train the request load predictor on the downloaded datasets.

```bash
cd ../
CUDA_VISIBLE_DEVICES=0 python offline_train.py --response_type 1 --use_prompt 1 --resample 1
```


## Reproduce Motivation Study

To reproduce the Azure LMaaS workload and ShareGPT load study, as presented in Figure 2, you can run the notebook `./motivation_study/1-data_distribution.ipynb` to analyze datasets in `../data`.

To reproduce the unbalanced timeline of two baseline request routing
algorithms (Least Request & Minimum Usage Routing), as presented in Figure 4, you can run the notebook `./motivation_study/2-load_balancing.ipynb` to analyze the provided benchmark results `./motivation_study/case_1` and `./motivation_study/case_2`. Alternatively, you can reproduce the 2 benchmark results with the same configurations by running the following commands:

```bash
# Start the vllm servers first (4 instances)
bash ./scripts/start_vllm_7b_4.sh

cd ../LLMServe
# case_1
python benchmark.py \
	--request_num "2800" \
	--model_name "meta-llama/Llama-2-7b-hf" \
	--result_dir "../results/cases/" \
	--load "ShareGPT" \
	--load_dataset_path "../data/datasets/ShareGPT/cleaned.csv" \
	--workload "poisson" \
	--qps "9.5" \
	--num_instances "4" \
	--scheduler_policy "least_requests" \
	--scaler_policy "none" \
	--req_predictor_policy "ground_truth" \
	--seed 7 
# case_2
python benchmark.py \
	--request_num "2800" \
	--model_name "meta-llama/Llama-2-7b-hf" \
	--result_dir "../results/cases/" \
	--load "ShareGPT" \
	--load_dataset_path "../data/datasets/ShareGPT/cleaned.csv" \
	--workload "poisson" \
	--qps "9.5" \
	--num_instances "4" \
	--scheduler_policy "min_utilization" \
	--scaler_policy "none" \
	--req_predictor_policy "ground_truth" \
	--seed 7 
```


## Reproduce RQ1: Accuracy of Hierarchical Prediction

As presented in Table 1, the **Request Workload Prediction** component achieves 7.74%, 8.45%, 4.15% and 4.30% of MAPE on Azure-code prompts, Azure-code responses, Azure-conv prompts and Azure-conv responses. To reproduce this, you can run the `./RQ1/workload_predictor/mLSTM_predict.py`:

```bash
cd ./RQ1/workload_predictor
python mLSTM_predict.py
```

As presented in Table 2, the **Request Load (Lengths) Prediction** component achieves 78.25 tokens of MAE and 77.95% of Acc-100 on ShareGPT dataset. The reproduction of these results can be found in the evaluation output of the load predictor model training script:

```bash
cd ../
CUDA_VISIBLE_DEVICES=0 python offline_train.py --response_type 1 --use_prompt 1 --resample 1
```


## Reproduce RQ2: Effectiveness for Instance Scaling

To reproduce the instance scaling results under 2 workloads using 7 auto-scaling strategies, as presented in Figure 8, you can run the `./RQ2/RQ2_instance_scaling.py` to analyze the provided RQ2 benchmark results (See [Prerequisite](./Reproducibility.md#prerequisite) for download instructions):

```bash
cd ./RQ2
cp -r ../PreServe_benchmark_results/RQ2/case_* .
python RQ2_instance_scaling.py
```

Alternatively, you can locally reproduce the eight benchmark results by running the following script:

```bash
# You should clear all cases in ./RQ2 and ../results/cases first
cd ../scripts
./start_vllm_7b_8.sh
./benchmark_RQ2.sh
```

This script runs the scaling simulation benchmark with 14 different configurations, each taking approximately 2 hours. It will generate 14 case directories in `../results/cases`. Before using them as benchmark results, you need to preprocess the data and copy them to the appropriate location:

```bash
cd ../results/
python result_analysis_metrics.py
cd ../experiments/RQ2
cp -r ../results/cases/case_* .
python RQ2_instance_scaling.py
```


## Reproduce RQ3: Effectiveness for Request Scheduling

The request scheduling experiments using LLaMA-2-7B are illustrated in figure `./RQ3/RQ3_request_routing.pdf`:

![Request Scheduling](./RQ3/RQ3_request_routing.png)

To reproduce these results, you can run the `./RQ3/RQ3_request_routing.py` to analyze the provided RQ3 benchmark results (See [Prerequisite](./Reproducibility.md#prerequisite) for download instructions):

```bash
cd ./RQ3
cp -r ../PreServe_benchmark_results/RQ3_7b ./LLaMA_2_7B

python RQ3_request_routing.py
```

Alternatively, you can locally reproduce the benchmark results by running the following script:

```bash
cd ../scripts
./start_vllm_7b_4.sh
./benchmark_RQ3_7b.sh
```

You may need to run the script multiple times to minimize the impact of variance. See RQ2 for further instructions on using these locally generated benchmark results.


## Reproduce RQ4: LMaaS Management Efficiency

To reproduce the LMaaS management overhead experiment, as presented in Figure 10, you can select any benchmark result from RQ3 to analyze its efficiency. In our case, we use the first benchmark result for LLaMA-2-7B:

```bash
cd ./RQ4
cp -r ../PreServe_benchmark_results/RQ3_7b/case_317 .
python RQ4-efficiency.py  # you may need to change the case-id
```
