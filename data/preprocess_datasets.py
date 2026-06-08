from transformers import AutoTokenizer
from huggingface_hub import login
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import argparse
import json
import os


def add_parser_arg(parser):
    parser.add_argument(
        "--datasets_path",
        type=str,
        nargs="?",
        help="The path of datasets",
        default="./datasets",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        nargs="?",
        help="The name of datasets",
        default="ShareGPT",
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        nargs="?",
        help="The file of datasets",
        default="ShareGPT.json",
    )

    parser.add_argument(
        "--tokenizer_name",
        type=str,
        nargs="?",
        help="The name of used tokenizer",
        default="meta-llama/Meta-Llama-3-8B",
    )
    parser.add_argument(
        "--num_requests",
        type=int,
        help="The max number of preprocessed requests",
        default=0,
    )
    parser.add_argument(
        "--min_input_tokens",
        type=int,
        help="The min number of input tokens",
        default=16,
    )
    parser.add_argument(
        "--min_out_tokens", type=int, help="The min number of output tokens", default=16
    )
    parser.add_argument(
        "--max_input_tokens",
        type=int,
        help="The max number of input tokens",
        default=512,
    )
    parser.add_argument(
        "--max_out_tokens",
        type=int,
        help="The max number of output tokens",
        default=512,
    )
    parser.add_argument(
        "--min_total_tokens",
        type=int,
        help="The min number of total tokens",
        default=None,
    )
    parser.add_argument(
        "--max_total_tokens",
        type=int,
        help="The max number of total tokens",
        default=None,
    )
    parser.add_argument(
        "--token_distribution_plot",
        type=int,
        help="Whether to plot the distribution of tokens. 0: do not plot, 1: process and plot, 2: only plot",
        default=1,
    )

    parser.add_argument(
        "--hugging_face_token",
        type=str,
        nargs="?",
        help="The hugging face login token",
        default=None,
    )


def plot_token_distribution(args, prompt_lens, response_lens):
    print("Statictics of token distribution")
    print(f"Number of requests: {len(prompt_lens)}")
    print(
        f"Prompt: min: {min(prompt_lens)}, max: {max(prompt_lens)}, mean: {sum(prompt_lens) / len(prompt_lens)}"
    )
    print(
        f"Response: min: {min(response_lens)}, max: {max(response_lens)}, mean: {sum(response_lens) / len(response_lens)}"
    )
    total_lens = [a + b for a, b in zip(prompt_lens, response_lens)]
    print(f"Total: min: {min(total_lens)}, max: {max(total_lens)}, mean: {sum(total_lens) / len(total_lens)}")

    print("Plotting token distribution")
    datasets_path, dataset_name = args.datasets_path, args.dataset_name
    min_input_tokens, min_out_tokens, max_input_tokens, max_out_tokens = (
        args.min_input_tokens,
        args.min_out_tokens,
        args.max_input_tokens,
        args.max_out_tokens,
    )
    file_postfix = f"_{min_input_tokens}_{min_out_tokens}_{max_input_tokens}_{max_out_tokens}"

    if args.min_total_tokens is not None or args.max_total_tokens is not None:
        file_postfix += f"_{args.min_total_tokens}_{args.max_total_tokens}"

    input_plot_path = os.path.join(
        datasets_path,
        os.path.join(
            dataset_name,
            f"input_tokens_distribution{file_postfix}.png",
        ),
    )
    output_plot_path = os.path.join(
        datasets_path,
        os.path.join(
            dataset_name,
            f"output_tokens_distribution{file_postfix}.png",
        ),
    )
    plt.hist(prompt_lens, bins=100)
    plt.savefig(input_plot_path)
    plt.close()
    plt.hist(response_lens, bins=100)
    plt.savefig(output_plot_path)
    plt.close()


def process_shareGPT(args):
    datasets_path, dataset_name, dataset_file = (
        args.datasets_path,
        args.dataset_name,
        args.dataset_file,
    )
    raw_file = os.path.join(datasets_path, os.path.join(dataset_name, dataset_file))
    print(f"Processing data file {raw_file}")
    num_requests = args.num_requests
    if num_requests:
        print(f"Max number of requests: {num_requests}")
    else:
        print(f"Max number of requests not set")

    min_input_tokens, min_out_tokens, max_input_tokens, max_out_tokens = (
        args.min_input_tokens,
        args.min_out_tokens,
        args.max_input_tokens,
        args.max_out_tokens,
    )
    file_postfix = f"_{min_input_tokens}_{min_out_tokens}_{max_input_tokens}_{max_out_tokens}"
    if args.min_total_tokens is not None or args.max_total_tokens is not None:
        file_postfix += f"_{args.min_total_tokens}_{args.max_total_tokens}"
    min_total_tokens = args.min_total_tokens if args.min_total_tokens else min_input_tokens + min_out_tokens
    max_total_tokens = args.max_total_tokens if args.max_total_tokens else max_input_tokens + max_out_tokens
    
    prompts = []
    completions = []
    prompt_lens = []
    response_lens = []
    print(tokenizer.name_or_path)
    with open(raw_file) as f:
        file = json.load(f)
        for data in tqdm(file):
            if len(data["conversations"]) >= 2:
                prompt = data["conversations"][0]["value"]
                completion = data["conversations"][1]["value"]
                prompt_token_ids = tokenizer(prompt).input_ids
                completion_token_ids = tokenizer(completion).input_ids
                
                if (
                    len(prompt_token_ids) >= min_input_tokens
                    and len(completion_token_ids) >= min_out_tokens
                    and len(prompt_token_ids) <= max_input_tokens
                    and len(completion_token_ids) <= max_out_tokens
                    and len(prompt_token_ids) + len(completion_token_ids) >= min_total_tokens
                    and len(prompt_token_ids) + len(completion_token_ids) <= max_total_tokens
                ):
                    prompts.append(data["conversations"][0]["value"])
                    completions.append(data["conversations"][1]["value"])
                    prompt_lens.append(len(prompt_token_ids))
                    response_lens.append(len(completion_token_ids))
            if num_requests and len(prompts) > num_requests:
                break

    print(f"Number of filtered requests: {len(prompts)}")

    if args.token_distribution_plot:
        plot_token_distribution(args, prompt_lens, response_lens)

    cleaned_data = {
        "prompts": prompts,
        "completions": completions,
        "prompt_lens": prompt_lens,
        "response_lens": response_lens,
    }

    cleaned_file_path = os.path.join(
        datasets_path,
        os.path.join(
            dataset_name,
            f"cleaned_{file_postfix}.csv",
        ),
    )
    print(
        f"Saving cleaned data to {cleaned_file_path}",
    )
    df = pd.DataFrame(cleaned_data)
    df.to_csv(cleaned_file_path, index=False)

    cleaned_file_path = os.path.join(
        datasets_path, os.path.join(dataset_name, f"cleaned.csv")
    )
    print(
        f"Saving cleaned data to {cleaned_file_path}",
    )
    df.to_csv(cleaned_file_path, index=False)

    return cleaned_file_path


def preprocess_data(args):
    if args.dataset_name == "ShareGPT":
        return process_shareGPT(args)
    elif args.dataset_name == "LMSYS-Chat-1M":
        return process_LMSYS(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_parser_arg(parser)
    args = parser.parse_args()

    if args.hugging_face_token is not None:
        login(token=args.hugging_face_token)

    datasets_path, dataset_name, dataset_file = (
        args.datasets_path,
        args.dataset_name,
        args.dataset_file,
    )
    raw_file = os.path.join(datasets_path, os.path.join(dataset_name, dataset_file))

    min_input_tokens, min_out_tokens, max_input_tokens, max_out_tokens = (
        args.min_input_tokens,
        args.min_out_tokens,
        args.max_input_tokens,
        args.max_out_tokens,
    )
    file_postfix = f"_{min_input_tokens}_{min_out_tokens}_{max_input_tokens}_{max_out_tokens}"
    if args.min_total_tokens is not None or args.max_total_tokens is not None:
        file_postfix += f"_{args.min_total_tokens}_{args.max_total_tokens}"

    cleaned_file_path = os.path.join(
        datasets_path,
        os.path.join(dataset_name, f"cleaned_{file_postfix}"),
    )

    if args.token_distribution_plot == 2 and os.path.exists(cleaned_file_path):
        df = pd.read_csv(cleaned_file_path)
        plot_token_distribution(args, df["prompt_lens"], df["response_lens"])
    else:
        print("Using tokenizer", args.tokenizer_name)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        preprocessed_file = preprocess_data(args)
