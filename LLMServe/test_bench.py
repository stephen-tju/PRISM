import os


def generate_command(
    request_num: int = 100,
    dataset: str = "ShareGPT",
    datasets_path: str = "../data/datasets/",
    dataset_file: str = "cleaned.csv",
    seed: int = 42,
    random_prompt_lens_mean: int = 256,
    random_prompt_lens_range: int = 16,
    random_response_lens_mean: int = 256,
    random_response_lens_range: int = 16,
    workload: str = "uniform",
    workloads_path: str = "../data/workloads/",
    qps: float = 1.0,
    scale: float = 1.0,
    coefficient_variation: float = 0.0,
    model_name: str = "meta-llama/Meta-Llama-3-8B",
    result_dir: str = "../results/",
    backend_framework: str = "vLLM",
    ignore_eos: bool = True,
    stream: bool = True,
    host: str = "localhost",
    port: int = 8000,
    temperature: float = 0.0,
):
    command = (
        f"python benchmark.py "
        f"--request_num {request_num} "
        f"{'--dataset ' + dataset if dataset else ''} "
        f"--datasets_path {datasets_path} "
        f"--dataset_file {dataset_file} "
        f"--seed {seed} "
        f"--random_prompt_lens_mean {random_prompt_lens_mean} "
        f"--random_prompt_lens_range {random_prompt_lens_range} "
        f"--random_response_lens_mean {random_response_lens_mean} "
        f"--random_response_lens_range {random_response_lens_range} "
        f"--workload {workload} "
        f"--workloads_path {workloads_path} "
        f"--qps {qps} "
        f"--scale {scale} "
        f"--coefficient_variation {coefficient_variation} "
        f"--model_name {model_name} "
        f"--result_dir {result_dir} "
        f"--backend_framework {backend_framework} "
        f"{'--ignore_eos' if ignore_eos else ''} "
        f"--max_tokens {max_tokens} "
        f"{'--stream' if stream else ''} "
        f"--host {host} "
        f"--port {port} "
        f"--temperature {temperature} "
    ).strip()
    return command


# if __name__ == "__main__":
#     try:
#         command = generate_command()
#         result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)

#         print(result.stdout)
#     except subprocess.CalledProcessError as e:
#         print(f"An error occurred while executing command: {command}")
#         print(e)
