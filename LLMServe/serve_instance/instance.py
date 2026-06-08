import time
import aiohttp
import json
import sys
import traceback
import re

from LLMServe.logger import init_logger
from LLMServe.util import get_tok_id_lens, remove_prefix
from LLMServe.util import get_openai_metrics_text, parse_openai_metrics_single, parse_openai_metrics_TTFT_acc, parse_openai_metrics_req_acc
from LLMServe.serve_instance.lookahead import Lookahead

logger = init_logger()

# Constant (a * predit_len + b would be better)
PREDICT_EXTEND = 30


class Instance:
    def __init__(self, instance_id, instance_config, scheduler_config):
        self.instance_id = instance_id
        self.request_num = 0
        
        self.incoming_prefill_tokens = 0
        self.incoming_decode_tokens = 0
        self.incoming_all_tokens = 0
        self.expected_token_usage = 0
        self.current_token_usage = 0
        self.imminent_token_usage = 0

        self.instance_config = instance_config
        logger.debug(
            f"Instance {self.instance_id} created with config: {self.instance_config}"
        )

        # float type (float division of load metrics)
        self.max_batch_size = min(self.instance_config["max_model_len"], self.instance_config["max_num_batched_tokens"])
        self.max_num_seqs = self.instance_config["max_num_seqs"]

        self.gpu_utilization = 0
        self.gpu_memory_utilization = 0

        self.lookahead = Lookahead(instance_id=instance_id, max_context_length=int(self.max_batch_size))



    async def request_inference(self, request_id, request, expected_output_len):
        prompt_len = int(request[1])
        output_len = expected_output_len
        self.request_num += 1
        self.incoming_prefill_tokens += prompt_len
        self.incoming_decode_tokens += output_len
        # self.incoming_all_tokens += (prompt_len + output_len) 
        self.expected_token_usage += (prompt_len + output_len)
        # self.imminent_token_usage += prompt_len

        # logger.debug(f"Instance {self.instance_id} received request with length {request[1]}. Current request_num: {self.request_num}")
        # logger.debug(f"Instance {self.instance_id} received request: \"{request}\"")
        result = await self.vllm_generate_call(
            self.instance_id, request_id, request, output_len, self.instance_config,
            endpoint="/v1/completions",
        )

        self.request_num -= 1
        self.expected_token_usage -= (prompt_len + output_len)
        if result["request_finished"]:
            self.incoming_decode_tokens -= 1
            # self.incoming_all_tokens -= 1
            # self.current_token_usage -= (prompt_len + output_len)
            # self.imminent_token_usage -= (prompt_len + output_len + 20)

        else: # failed, usually request abort
            if result["generated_tokens"] == 0: # failed at first token
                self.incoming_prefill_tokens -= prompt_len
                self.incoming_decode_tokens -= output_len
                # self.incoming_all_tokens -= (prompt_len + output_len)
                # self.imminent_token_usage -= prompt_len
            else:
                self.incoming_decode_tokens -= (output_len - result["generated_tokens"] + 1)
                # self.incoming_all_tokens -= (output_len - result["generated_tokens"] + 1)
                # self.current_token_usage -= (prompt_len + result["generated_tokens"])
                # self.imminent_token_usage -= (prompt_len + result["generated_tokens"] + 20)
    
        return result

    def prefill_done_callback(self, prompt_len):
        self.incoming_prefill_tokens -= prompt_len
        # self.incoming_all_tokens -= prompt_len
        # self.current_token_usage += (prompt_len + 1)
        # self.imminent_token_usage += 21  # TODO: should have quantative ratio on decoding prediction
    
    def decode_done_callback(self):
        self.incoming_decode_tokens -= 1
        # self.incoming_all_tokens -= 1
        # self.current_token_usage += 1
        # self.imminent_token_usage += 1

    def decode_excess_callback(self):
        self.incoming_decode_tokens += PREDICT_EXTEND
        # self.incoming_all_tokens += PREDICT_EXTEND
        self.expected_token_usage += PREDICT_EXTEND
        # self.imminent_token_usage += PREDICT_EXTEND


    def get_instance_id(self):
        return self.instance_id

    def get_instance_request_num(self):
        return self.request_num
    
    # async def get_instance_GPU_cache_usage_perc(self):
    #     metrics = await self.get_openai_metrics()
    #     if metrics:
    #         return metrics["vllm:gpu_cache_usage_perc"]
    #     return 0

    def get_instance_expected_token_usage(self):
        return self.expected_token_usage

    def get_instance_current_token_usage(self):
        return self.current_token_usage 

    def get_instance_incoming_prefill_tokens(self):
        return self.incoming_prefill_tokens

    def get_instance_incoming_decode_tokens(self):
        return self.incoming_decode_tokens

    def get_instance_incoming_all_tokens(self):
        return self.incoming_all_tokens

    def get_instance_imminent_token_usage(self):
        return self.imminent_token_usage

    def get_instance_prefill_load_0(self): # L1
        return self.incoming_prefill_tokens / self.max_batch_size

    def get_instance_expected_decode_load_0(self): # L2
        return self.incoming_decode_tokens / self.max_num_seqs

    def get_instance_load_0(self):
        L1 = self.incoming_prefill_tokens / 64
        L2 = self.incoming_decode_tokens / self.max_num_seqs
        # print("Overload: ", L1, L2)
        return max(L1, L2)

    def get_instance_load_1(self):
        L1 = self.incoming_prefill_tokens / 64
        L2 = self.incoming_decode_tokens / self.max_num_seqs
        return L1 + L2

    # This param should be decided by 'prefill time per token' : 'decode time per token' 
    def get_instance_load_3(self, param):
        return self.incoming_prefill_tokens * param + self.incoming_decode_tokens
    
    def get_instance_load_4(self, param):
        return max(self.incoming_prefill_tokens * param, self.incoming_decode_tokens)

    def get_instance_load_5(self, param):
        return self.incoming_prefill_tokens * param + self.expected_token_usage

    def update_gpu_memory_utilization(self, gpu_memory_utilization):
        self.gpu_memory_utilization = gpu_memory_utilization
    
    def update_gpu_utilization(self, gpu_utilization):
        self.gpu_utilization = 0.5 * self.gpu_utilization + 0.5 * gpu_utilization
    
    def get_instance_gpu_memory_utilization(self):
        return self.gpu_memory_utilization

    def get_instance_utilization(self):
        return 0.0 * self.gpu_utilization + 1.0 * self.gpu_memory_utilization 

    def get_instance_lookahead_max_tokens(self):
        return self.lookahead.get_lookahead_max_tokens()

    def get_instance_lookahead_200(self):
        return self.lookahead.get_lookahead_200()

    
    async def get_openai_metrics(self):
        metrics_text = await get_openai_metrics_text(self.instance_config['host'], self.instance_config['port'])
        # metrics = parse_openai_metrics(metrics_text)
        metrics = parse_openai_metrics_single(metrics_text)
        return metrics
    
    async def get_openai_metrics_TTFT_acc(self):
        metrics_text = await get_openai_metrics_text(self.instance_config['host'], self.instance_config['port'])
        ttft_acc = parse_openai_metrics_TTFT_acc(metrics_text)
        return ttft_acc

    async def get_openai_metrics_TTFT_req_acc(self):
        metrics_text = await get_openai_metrics_text(self.instance_config['host'], self.instance_config['port'])
        ttft_acc = parse_openai_metrics_TTFT_acc(metrics_text)
        req_acc = parse_openai_metrics_req_acc(metrics_text)
        return ttft_acc, req_acc

    

    async def vllm_generate_call(self, instance_id, request_id, request, request_output_len, instance_config, endpoint="/generate"):
        # tokenizer = AutoTokenizer.from_pretrained(instance_config["model_name"])
        prompt, prompt_len, response_len = (
            request[0],
            int(request[1]),
            request_output_len
        )
        timeout = aiohttp.ClientTimeout(total=4 * 60 * 60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            payload = {
                "model": instance_config["model_name"],
                "prompt": prompt,
                "temperature": instance_config["temperature"],
                "best_of": instance_config["best_of"],
                "max_tokens": response_len,
                # "stream": instance_config["stream"],
                "stream": True,
                "ignore_eos": True,
            }
            headers = {
                "Authorization": f"Bearer {instance_config['openai_api_key']}",
            }
            # vllm_request_id = None
            ttft = 0.0
            latency = 0.0
            generated_tokens = 0
            itl = []

            st = time.time()
            try:
                async with session.post(
                    f"http://{instance_config['host']}:{instance_config['port']}{endpoint}",
                    json=payload,
                    headers=headers,
                ) as resp:
                    if resp.status != 200:
                        logger.error(f"Error: {resp.status} {resp.reason}")
                        logger.error(await resp.text())
                        request_results = {
                            "request_id": request_id,
                            "instance_id": instance_id,
                            "request_finished": False,
                            "prompt_tokens": prompt_len,
                            "expected_tokens": response_len,
                            "generated_tokens": generated_tokens,
                            "record_time": time.time(),
                            "error": f"Error: {resp.status} {resp.reason}",
                        }
                        return request_results

                    else: # well recieved response
                        async for chunk_bytes in resp.content:
                            chunk_bytes = chunk_bytes.strip()
                            if not chunk_bytes:
                                continue

                            chunk = remove_prefix(chunk_bytes.decode("utf-8"),
                                                "data: ")
                            if chunk == "[DONE]":
                                latency = time.time() - st
                                self.lookahead.request_end(request_id)
                            else:
                                data = json.loads(chunk)
                                if data["choices"][0]["text"]:
                                    timestamp = time.time()
                                    # First token
                                    if ttft == 0.0:
                                        ttft = time.time() - st
                                        self.lookahead.request_first_token(request_id, prompt_len, response_len)
                                        self.prefill_done_callback(prompt_len) 
                                        # vllm_request_id = data["id"]
                                    else:
                                        self.lookahead.request_other_token(request_id, generated_tokens+1)
                                        self.decode_done_callback()
                                        if generated_tokens >= response_len:
                                            response_len += PREDICT_EXTEND
                                            self.decode_excess_callback()
                                    
                                    itl.append(timestamp)
                                    generated_tokens += 1

                    # end_reason = str(data["choices"][0]["finish_reason"]) + "_" + str(data["choices"][0]["stop_reason"])
                    request_results = {
                        # "vllm_request_id": vllm_request_id,
                        "request_id": request_id,
                        "instance_id": instance_id,
                        "request_finished": True,
                        "prompt_tokens": prompt_len,
                        "expected_tokens": response_len,
                        "generated_tokens": generated_tokens,
                        "TTFT": ttft,
                        "TBPT": (latency - ttft) / response_len,
                        "latency": latency,
                        "record_time": time.time(),
                        "itl": itl,
                        "error": "",  # end_reason,
                    }
                    # logger.info(generated_text)

            except Exception:
                self.lookahead.request_end(request_id)
                logger.error(f"Error: {traceback.format_exception(*sys.exc_info())}")
                request_results = {
                    # "vllm_request_id": vllm_request_id,
                    "request_id": request_id,
                    "instance_id": instance_id,
                    "request_finished": False,
                    "prompt_tokens": prompt_len,
                    "expected_tokens": response_len,
                    "generated_tokens": generated_tokens,
                    "record_time": time.time(),
                    "itl": itl,
                    "error": "".join(traceback.format_exception(*sys.exc_info())),
                }

        if generated_tokens != response_len:
            request_results["request_finished"] = False  # Groundtruth verify
        return request_results


    @staticmethod 
    def get_dummy_request_result(request_id, request):
        return {
            "request_id": request_id,
            "instance_id": -1,
            "request_finished": False,
            "prompt_tokens": int(request[1]),
            "expected_tokens": int(request[2]),
            "generated_tokens": 0,
            "record_time": time.time(),
            "error": "Dummy request result",
        }
