import os
import sys
import time
import json
import numpy as np
from datetime import datetime
import random
import pandas as pd
from LLMServe.logger import init_logger
from LLMServe.util import read_dataset

logger = init_logger()



class Workload(object):
    def __init__(self, request_config):
        self.trace = None
        self.request_id = 0
        self.last_request_time = 0.0

        self.request_config = request_config
        self.workload_name = request_config["workload"]
        self.workload_mode = request_config["workload_mode"]

        if self.workload_mode == "trace":
            logger.info(f"Loading workload trace from {request_config['workload_trace_path']}")
            self.split_trace_dataset(
                read_dataset(self.request_config["workload_trace_path"])
            )
            # logger.info(f"Workload trace loaded, total train:{len(self.train_trace)}, test:{len(self.trace)} requests")
            
            # test data for actual benchmark
            self.trace = self.sample_trace(
                trace=self.trace, 
                sampling_interval=request_config["workload_sampling_interval"])
            self.trace = self.timerange_trace(
                trace=self.trace,
                ratio_timerange=(request_config["workload_trace_range_L"], request_config["workload_trace_range_R"]))
            self.trace = self.timescale_trace(
                trace=self.trace,
                timescale=request_config["workload_timescale"])
    
            self.last_request_time = self.trace.at[0, "Timestamp"]
            end_time = request_config["benchmark_duration"]
            if end_time is not None:
                self.trace = self.trace[self.trace["Timestamp"] < end_time]

            if len(self.trace) > request_config["request_num"]:
                self.trace = self.trace.iloc[:request_config["request_num"], :]
            elif len(self.trace) < request_config["request_num"]:
                request_config["request_num"] = len(self.trace)

            # train data for workload predictor
            self.train_trace = self.sample_trace(
                trace=self.train_trace, 
                sampling_interval=request_config["workload_sampling_interval"])
            self.train_trace = self.timescale_trace(
                trace=self.train_trace,
                timescale=request_config["workload_timescale"])


        self.qps = request_config["qps"]
        self.coefficient_variation = request_config["coefficient_variation"]


    # Sample every [interval] requests, but keep the original order
    def sample_trace(self, trace, sampling_interval=1):
        trace = trace.iloc[::sampling_interval, :]
        trace.reset_index(drop=True, inplace=True)
        return trace
    
    # Speed up the trace simulation by a factor of [timescale]
    def timescale_trace(self, trace, timescale=1):
        trace["Timestamp"] = trace["Timestamp"] - trace.at[0, "Timestamp"]
        trace["Timestamp"] = trace["Timestamp"] / timescale
        return trace
    
    def timerange_trace(self, trace, ratio_timerange=(0.0, 1.0)):
        all_start_time, all_end_time = min(trace["Timestamp"]), max(trace["Timestamp"])
        start_time = all_start_time + ratio_timerange[0] * (all_end_time - all_start_time)
        end_time = all_start_time + ratio_timerange[1] * (all_end_time - all_start_time)
        trace = trace[(trace["Timestamp"] >= start_time) & (trace["Timestamp"] <= end_time)]
        trace.reset_index(drop=True, inplace=True)
        return trace
    
    
    def split_trace_dataset(self, all_trace):
        train_dataset_ratio = 0.5
        self.train_trace = all_trace.iloc[:int(len(all_trace) * train_dataset_ratio), :]
        self.trace = all_trace.iloc[int(len(all_trace) * train_dataset_ratio):, :]


    def get_train_trace(self):
        assert self.train_trace is not None
        return self.train_trace

    def get_trace(self):
        assert self.trace is not None
        return self.trace


    def get_request_time(self, request_id):
        assert request_id == self.request_id
        assert self.request_id < self.request_config["request_num"]

        if self.workload_mode == "trace":
            self.last_request_time = self.trace.at[self.request_id, "Timestamp"]
        else:
            self.last_request_time += self.get_random_wait_time()

        self.request_id += 1
        return self.last_request_time


    def get_random_wait_time(self):
        mean_time_between_requests = 1 / self.qps
        if self.workload_name== "uniform":
            return mean_time_between_requests
        elif self.workload_name== "gamma":
            variance = (self.coefficient_variation * mean_time_between_requests) ** 2
            shape = mean_time_between_requests ** 2 / variance
            return np.random.gamma(shape, variance / mean_time_between_requests)
        elif self.workload_name== "poisson":
            return np.random.exponential(mean_time_between_requests)
        else:
            logger.error("Unsupported workload: ", self.workload_name)
            return None 
