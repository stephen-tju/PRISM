import asyncio
import copy
import random
import time
from LLMServe.serve_instance.instance import Instance
from LLMServe.logger import init_logger
from LLMServe.util import load_instance_configuration
from LLMServe.global_scheduler.load_predictor.predictor import LoadPredictor

logger = init_logger()


class Scheduler:
    def __init__(self, scheduler_config, instance_config):
        self.num_instances = scheduler_config["num_instances"]
        # logger.info(f"Initializing instances.")

        instance_slots = load_instance_configuration(
            scheduler_config["instance_configurations"]
        )
        if len(instance_slots) < self.num_instances:
            raise ValueError("The initial number of instances exceeded the available slots in the configuration.")

        self.instance_slots = []
        for ins_id, ins_cfg in enumerate(instance_slots):
            detailed_config = copy.deepcopy(instance_config)
            detailed_config.update(ins_cfg)
            self.instance_slots.append(Instance(ins_id, detailed_config, scheduler_config))
        self.instances = self.instance_slots[:self.num_instances] 
        logger.info(f"Instances initialized, now using {len(self.instances)} / {len(self.instance_slots)} available slots.")

        # # self.scaling_lock = asyncio.Lock()
        # if scheduler_config["scaler_policy"] != "none":
        #     asyncio.create_task(self.scaler_monitor(scheduler_config["scaler_policy"], 300))

        self.max_model_len = max(instance_config["max_model_len"], instance_config["max_num_seqs"])

        self.last_selected_instance = 0
        self.scheduler_policy = scheduler_config["scheduler_policy"]
        self.scheduler_param = scheduler_config["scheduler_param"]
        self.req_predictor_policy = scheduler_config["req_predictor_policy"]
        logger.info(f"Schduler policy: {self.scheduler_policy} x {self.req_predictor_policy}")

        self.request_load_predictor = LoadPredictor(scheduler_config)

        self.prompt_tokens_acc = 0
        self.response_tokens_acc = 0


    def test_predictor(self, text):
        return self.request_load_predictor.predict(text)


    def get_num_instances(self):
        return self.num_instances
    
    def get_num_instance_slots(self):
        return len(self.instance_slots) 

    def get_prompt_tokens_acc(self):
        return self.prompt_tokens_acc

    def get_response_tokens_acc(self):
        return self.response_tokens_acc


    def get_instances_load_info(self):
        load_info = []
        for i, instance in enumerate(self.instances):
            load_info.append({
                # "instance_id": i,
                "request_num": instance.get_instance_request_num(),
                "incoming_prefill_tokens": instance.get_instance_incoming_prefill_tokens(),
                "incoming_decode_tokens": instance.get_instance_incoming_decode_tokens(),
                # "incoming_all_tokens": instance.get_instance_incoming_all_tokens(),
                "expected_token_usage": instance.get_instance_expected_token_usage(),
                "current_token_usage": instance.get_instance_current_token_usage(),
                "lookahead_max_tokens": instance.get_instance_lookahead_max_tokens(),
                'lookahead_200': instance.get_instance_lookahead_200(),
            })
        return load_info


    async def get_instances_openai_metrics(self):
        openai_metrics = []
        for i, instance in enumerate(self.instances):
            openai_metrics.append(
                await instance.get_openai_metrics()
            )
        return openai_metrics


    def get_round_robin_instance(self):
        self.last_selected_instance = (self.last_selected_instance + 1) % self.num_instances
        return self.instances[self.last_selected_instance]

    def get_random_instance(self):
        return self.instances[random.randint(0, self.num_instances - 1)]

    def get_least_loaded_instance(self):
        return min(self.instances, key=lambda instance: instance.get_instance_request_num())

    # def get_ground_truth_instance(self):
    #     # return min(self.instances, key=lambda instance: instance.get_instance_load_0())
    #     # return min(self.instances, key=lambda instance: instance.get_instance_expected_token_usage())
    #     # return min(self.instances, key=lambda instance: instance.get_instance_current_token_usage())
    #     # return min(self.instances, key=lambda instance: instance.get_instance_incoming_decode_tokens())
    #     # return min(self.instances, key=lambda instance: instance.get_instance_incoming_prefill_tokens())
    #     return min(self.instances, key=lambda instance: instance.get_instance_imminent_token_usage())

    def get_load_0_instance(self):
        return min(self.instances, key=lambda instance: instance.get_instance_load_0())

    def get_load_1_instance(self):
        return min(self.instances, key=lambda instance: instance.get_instance_load_1())

    def get_load_3_instance(self):
        return min(self.instances, key=lambda instance: instance.get_instance_load_3(self.scheduler_param))

    def get_load_4_instance(self):
        return min(self.instances, key=lambda instance: instance.get_instance_load_4(self.scheduler_param))

    def get_load_5_instance(self):
        return min(self.instances, key=lambda instance: instance.get_instance_load_5(2))

    def update_gpu_utilizations(self, gpu_utilizations):
        for i, instance in enumerate(self.instances):
            instance.update_gpu_utilization(gpu_utilizations[i])

    def update_gpu_memory_utilizations(self, gpu_memory_utilizations):
        for i, instance in enumerate(self.instances):
            instance.update_gpu_memory_utilization(gpu_memory_utilizations[i])

    def get_min_utilization_instance(self):
        return min(self.instances, key=lambda instance: instance.get_instance_utilization())

    def get_min_gpu_memory_utilization_instance(self):
        return min(self.instances, key=lambda instance: instance.get_instance_gpu_memory_utilization())


    # Scaler policy: monitor_scheduler_load
    def get_instances_avg_load(self):
        # return sum([instance.get_instance_request_num() for instance in self.instances]) / self.num_instances
        return sum([instance.get_instance_load_5(2) for instance in self.instances]) / self.num_instances

    def get_instances_avg_mem(self):
        return sum([instance.get_instance_gpu_memory_utilization() for instance in self.instances]) / self.num_instances

    def get_instances_avg_lookahead_max(self):
        return sum([instance.get_instance_lookahead_max_tokens() for instance in self.instances]) / self.num_instances


    async def handle_request(self, request_id, request):
        # logger.info(f"Prompt Length: {int(request[1])}, Response Length: {int(request[2])}")
        req_predictor_start = time.time()
        if self.req_predictor_policy == "load_predictor":
            expected_output_len = self.request_load_predictor.predict(request[0])
            if expected_output_len + int(request[1]) > self.max_model_len:
                expected_output_len = int(self.max_model_len) - int(request[1])
        elif self.req_predictor_policy == "ground_truth":
            expected_output_len = int(request[2])
        else:
            raise ValueError("Invalid request predictor policy")
        req_predictor_end = time.time()

        if self.scheduler_policy == "load_0":
            instance = self.get_load_0_instance()
        elif self.scheduler_policy == "load_1":
            instance = self.get_load_1_instance()
        elif self.scheduler_policy == "load_2":
            instance = self.get_least_loaded_instance()
        elif self.scheduler_policy == "load_3":
            instance = self.get_load_3_instance()
        elif self.scheduler_policy == "load_4":
            instance = self.get_load_4_instance()
        elif self.scheduler_policy == "load_5":
            instance = self.get_load_5_instance()
        elif self.scheduler_policy == "preserve":
            instance = self.get_load_5_instance()
        elif self.scheduler_policy == "least_requests":
            instance = self.get_least_loaded_instance()
        elif self.scheduler_policy == "round_robin":
            instance = self.get_round_robin_instance()
        elif self.scheduler_policy == "random":
            instance = self.get_random_instance()
        elif self.scheduler_policy == "least_utilization":
            instance = self.get_min_utilization_instance()
        elif self.scheduler_policy == "least_gpu_memory_utilization":
            instance = self.get_min_gpu_memory_utilization_instance()
        else:
            raise ValueError("Invalid scheduler policy")
    
        if instance == None: # dummy failed result
            logger.info(f"Request {request_id} dumped to invalid instance")
            return Instance.get_dummy_request_result(request_id, request)

        self.prompt_tokens_acc += int(request[1])
        self.response_tokens_acc += int(request[2])

        result = await instance.request_inference(request_id, request, expected_output_len)
        result["req_predictor_time"] = req_predictor_end - req_predictor_start
        return result


    def scale_instances_to(self, num_instances):
        if num_instances > len(self.instance_slots) or num_instances < 1:
            raise ValueError(f"The number of instances {num_instances} exceeded the available slots {len(self.instance_slots)} in the configuration.")

        self.instances = self.instance_slots[:num_instances]
        self.num_instances = num_instances
