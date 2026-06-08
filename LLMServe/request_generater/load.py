import pandas as pd
import random
import numpy as np
from scipy.stats import zipf
from transformers import AutoTokenizer
from LLMServe.logger import init_logger
from LLMServe.util import read_dataset, merge_lists_by_ratio

logger = init_logger()



class Load(object):
    def __init__(self, request_config, tokenizer_name, trace=None):
        self.load_name = request_config["load"]
        self.load_mode = request_config["load_mode"]
        # self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer = None
        self.data = None
        self.request_id = 0

        if self.load_mode == "prompt_dataset":
            df = read_dataset(request_config["load_dataset_path"])
            # Workload trace size would truncate actual #req, but Load data would re-sample to match #req
            df = df.sample(
                n=request_config["request_num"], 
                replace=request_config["request_num"] > len(df), 
                random_state=request_config["seed"]
            ).reset_index(drop=True)
            # [prompt, _response, prompt_len, response_len]
            self.data = list(df.to_records(index=False))


        elif self.load_mode == "workload_trace_sample":
            df = read_dataset(request_config["load_dataset_path"])
            # Workload trace size would truncate actual #req, but Load data would re-sample to match #req
            df = df.sample(
                n=request_config["request_num"], 
                replace=request_config["request_num"] > len(df), 
                random_state=request_config["seed"]
            ).reset_index(drop=True)

            # [Timestamp, Request tokens, Response tokens] -> [prompt, _response, prompt_len, response_len]
            prompt_lens = df["Request tokens"].tolist()
            response_lens = df["Response tokens"].tolist()
            # prompts = self.gen_random_prompts_by_lengths(
            #     self.tokenizer, prompt_lens, vocab_ids_to_exclude=self.tokenizer.all_special_ids
            # )
            prompts = self.gen_random_prompts_by_lengths_simple(prompt_lens)
            _responses = [None] * len(response_lens)
            self.data = list(zip(prompts, _responses, prompt_lens, response_lens))


        elif self.load_mode == "random_generated":
            logger.info(f"Prompt length (mean={request_config['random_prompt_lens_mean']}, range={request_config['random_prompt_lens_range']})")
            logger.info(f"Response length (mean={request_config['random_response_lens_mean']}, range={request_config['random_response_lens_range']})")
            if request_config["dataset"] in ["uniform_mixing"]:
                # self.data = self.get_random_mixing_prompts(request_config)
                raise NotImplementedError("Mixing prompts not implemented")
            else:
                self.data = self.get_random_requests(request_config)


        elif self.load_mode == "workload_trace":
            assert trace is not None
            # [Timestamp, Request tokens, Response tokens] -> [prompt, _response, prompt_len, response_len]
            prompt_lens = trace["Request tokens"].tolist()
            response_lens = trace["Response tokens"].tolist()
            # prompts = self.gen_random_prompts_by_lengths(
            #     self.tokenizer, prompt_lens, vocab_ids_to_exclude=self.tokenizer.all_special_ids
            # )
            prompts = self.gen_random_prompts_by_lengths_simple(prompt_lens)
            _responses = [None] * len(response_lens)
            self.data = list(zip(prompts, _responses, prompt_lens, response_lens))

        else:
            raise ValueError(f"Unknown load mode: {self.load_mode}")


    def get_request(self, request_id):
        assert request_id == self.request_id

        # dataset_index = self.request_id % len(self.data)
        dataset_index = self.request_id
        prompt, _response, prompt_len, response_len = self.data[dataset_index]
        self.request_id += 1
        return prompt, prompt_len, response_len


    def get_random_requests(self, request_config):
        prompt_lens = self.gen_random_lengths(
            distribution=request_config["load"],
            len_mean=request_config["random_prompt_lens_mean"],
            len_range=request_config["random_prompt_lens_range"],
            request_num=request_config["request_num"],
        )
        response_lens = self.gen_random_lengths(
            distribution=request_config["load"],
            len_mean=request_config["random_response_lens_mean"],
            len_range=request_config["random_response_lens_range"],
            request_num=request_config["request_num"],
        )
        # prompts = self.gen_random_prompts_by_lengths(
        #     tokenizer=self.tokenizer,
        #     lens=prompt_lens,
        #     vocab_ids_to_exclude=self.tokenizer.all_special_ids,
        # )
        prompts = self.gen_random_prompts_by_lengths_simple(prompt_lens)
        _responses = [None] * response_lens
        return list(zip(prompts, _responses, prompt_lens, response_lens))


    # def get_random_mixing_prompts(self, request_config, tokenizer):
    #     request_num = request_config["request_num"]
    #     assert tokenizer is not None
    #     ratio = request_config["mixing_propotion"]
    #     prompts_part1, prompt_lens_part1 = gen_random_prompts_with_lens(
    #         tokenizer,
    #         distribution="uniform",
    #         len_mean=request_config["random_prompt_lens_mean"],
    #         len_range=request_config["random_prompt_lens_range"],
    #         request_num=int(request_num * ratio),
    #         vocab_ids_to_exclude=tokenizer.all_special_ids,
    #     )
    #     response_lens_part1 = get_random_lengths(
    #         distribution="uniform",
    #         len_mean=request_config["random_response_lens_mean"],
    #         len_range=request_config["random_response_lens_range"],
    #         request_num=int(request_num * ratio),
    #     )

    #     prompts_part2, prompt_lens_part2 = gen_random_prompts_with_lens(
    #         tokenizer,
    #         distribution="uniform",
    #         len_mean=request_config["random_response_lens_mean"],
    #         len_range=request_config["random_response_lens_range"],
    #         request_num=int(request_num * (1.0 - ratio)),
    #         vocab_ids_to_exclude=tokenizer.all_special_ids,
    #     )

    #     response_lens_part2 = get_random_lengths(
    #         distribution="uniform",
    #         len_mean=request_config["random_prompt_lens_mean"],
    #         len_range=request_config["random_prompt_lens_range"],
    #         request_num=int(request_num * (1.0 - ratio)),
    #     )

    #     # Shuffle the prompts
    #     # merged_list = [item for pair in zip(list1, list2) for item in pair]
    #     prompts = merge_lists_by_ratio(prompts_part1, prompts_part2, ratio)
    #     prompt_lens = merge_lists_by_ratio(prompt_lens_part1, prompt_lens_part2, ratio)
    #     response_lens = merge_lists_by_ratio(response_lens_part1, response_lens_part2, ratio)
    #     # prompts = [item for pair in zip(prompts_part1, prompts_part2) for item in pair]
    #     # prompt_lens = [item for pair in zip(prompt_lens_part1, prompt_lens_part2) for item in pair]
    #     # response_lens = [item for pair in zip(response_lens_part1, response_lens_part2) for item in pair]

    #     none_list = [None] * len(prompts)
    #     return list(zip(prompts, none_list, prompt_lens, response_lens))


    @staticmethod
    def gen_random_lengths(distribution, len_mean, len_range, request_num):
        if distribution == "uniform":
            if len_range == 0:
                return [len_mean for _ in range(request_num)]

            low = len_mean - (len_range // 2)
            high = len_mean + (len_range // 2)
            response_lens = list(
                map(lambda _: random.randint(low, high), range(request_num))
            )
        elif distribution == "exponential":
            response_lens = [
                min(round(s), len_range)
                for s in np.random.exponential(scale=len_mean, size=request_num)
            ]
        elif distribution == "capped_exponential":
            response_lens = []
            while len(response_lens) < request_num:
                sample = round(np.random.exponential(scale=len_mean))
                if sample <= len_range and sample >= 1:
                    response_lens.append(sample)
        elif distribution == "zipf":
            rank = np.arange(1, len_mean * 2)
            if len_mean == 1024 and len_range == 6144:
                alpha = 1.0005
            elif len_mean == 512 and len_range == 6144:
                alpha = 1.15
            elif len_mean == 256 and len_range == 6144:
                alpha = 1.5
            elif len_mean == 128 and len_range == 6144:
                alpha = 2.0
            else:
                alpha = 1.0
            probabilities = zipf.pmf(rank, alpha)
            probabilities /= np.sum(probabilities)
            response_lens = np.random.choice(
                np.arange(1, len_mean * 2), size=request_num, p=probabilities
            )
        else:
            raise ValueError(f"unknown distribution {distribution}")

        scaling_factor = len_mean / np.mean(response_lens)
        response_lens = np.ceil(np.array(response_lens) * scaling_factor).astype(int)
        if distribution == "zipf":
            response_lens = [
                response_len if response_len <= len_range else len_range
                for response_len in response_lens
            ]
        elif distribution == "uniform":
            capped_response_lens = []
            for response_len in response_lens:
                if response_len < low:
                    capped_response_lens.append(low)
                elif response_len > high:
                    capped_response_lens.append(high)
                else:
                    capped_response_lens.append(response_len)
            response_lens = capped_response_lens
        else:
            response_lens = [
                response_len if response_len <= len_range else len_range
                for response_len in response_lens
            ]
        response_lens = [int(x) for x in response_lens]

        return response_lens


    @staticmethod
    def gen_random_prompts_by_lengths(
        tokenizer, lens, vocab_ids_to_exclude=[]
    ):
        assert tokenizer is not None
        prompts = []
        for l in lens:
            prompt_token_ids = [random.randint(10, tokenizer.vocab_size) for _ in range(l)]
            prompt_text = tokenizer.decode(prompt_token_ids)

            # Because tokens do not map 1:1 to words, sometimes we get more tokens than desired.
            # This removes the additional tokens by tokenizing the prompt and cutting off additional tokens.
            # Confusingly, it works with a single iteration per prompt.
            encoded = tokenizer(prompt_text)["input_ids"]
            # assert len(encoded) == l, f"Expected prompt to contain exactly {l} tokens, got {len(encoded)=}"
            if len(encoded) > l:
                encoded = encoded[:l]
            decoded = tokenizer.decode(encoded)
            prompts.append(decoded)
        
        return prompts


    @staticmethod
    def gen_random_prompts_by_lengths_simple(lens):
        word = "hello"
        prompts = []
        for l in lens:
            prompt_text = " ".join([word for _ in range(l-1)])  # initial token
            prompts.append(prompt_text)
        return prompts
        