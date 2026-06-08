import os
import sys
import time
import json
import numpy as np
from datetime import datetime
import random
import pandas as pd
import matplotlib.pyplot as plt
from .load import Load
from .workload import Workload
from LLMServe.logger import init_logger


logger = init_logger()


class Generator:
    def __init__(self, request_config, model_name=None):
        self.request_config = request_config

        self.workload = Workload(request_config) # Might change request_config['request_num']
        logger.info(f"Using workload {request_config['workload']} ({request_config['workload_mode']}), actual #req = {request_config['request_num']}")

        self.load = Load(
            request_config=request_config, 
            tokenizer_name=model_name,
            trace=self.workload.get_trace() if request_config["load"] == "workload_trace" else None
        )
        logger.info(f"Using load {request_config['load']} ({request_config['load_mode']}), #req = {request_config['request_num']}")

        self.request_id = 0


    def get_request(self):
        sleep_time = self.workload.get_request_time(self.request_id)  
        prompt, prompt_len, response_len = self.load.get_request(self.request_id)
        self.request_id += 1
        return prompt, prompt_len, response_len, sleep_time


    def generate_requests(self):
        requests = []
        request_prompt_lens = []
        request_response_lens = []
        for _ in range(self.request_config["request_num"]):
            request = self.get_request()
            if request is not None:
                requests.append(request)
                request_prompt_lens.append(int(request[1]))
                request_response_lens.append(int(request[2]))
        
        plt.figure(figsize=(10, 6))
        plt.hist(request_prompt_lens, bins=10, alpha=0.5, label='Request Prompt Lengths')
        plt.hist(request_response_lens, bins=10, alpha=0.5, label='Request Response Lengths')
        plt.title('Distribution of Request Prompt and Response Lengths')
        plt.xlabel('Length')
        plt.ylabel('Frequency')
        plt.legend(loc='upper right')
        plt.savefig("../results/generated.png")
        
        return requests


    def get_workload_train_trace(self):
        return self.workload.get_train_trace()
    
    def get_workload_test_trace(self):
        return self.workload.get_trace()
        