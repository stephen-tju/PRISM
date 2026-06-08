import asyncio
from sklearn.metrics import mean_absolute_error
import numpy as np
from LLMServe.logger import init_logger
from LLMServe.global_scheduler.workload_predictor.predictor import WorkloadPredictor

logger = init_logger()


# COLD_START_TIME = 20
COLD_START_TIME = {
    "Azure_code_peak": 20,
    "Azure_conv_peak": 30,
}
SCALE_FREEZE_TIME = {
    "Azure_code_peak": 60,
    "Azure_conv_peak": 60,
}

class Scaler:
    def __init__(self, scheduler, scheduler_config, workload_name):
        self.scheduler = scheduler
        self.policy = scheduler_config["scaler_policy"]
        self.interval = scheduler_config["scaler_interval"]
        self.workload_name = workload_name

        self.instances = scheduler.instances
        self.instance_slots = scheduler.instance_slots
        self.num_instances = scheduler.get_num_instances()
        self.num_instance_slots = scheduler.get_num_instance_slots()

        self.scaling_lock = asyncio.Lock()
        self.monitor_tasks = []
        self.monitor_stop_event = asyncio.Event()

        self.vars = {}
        self.instance_slots_ttft = [0.0] * self.num_instance_slots  # for TTFT_threshold
        self.cold_start_time = COLD_START_TIME[workload_name]
        self.scale_freeze_time = SCALE_FREEZE_TIME[workload_name]


    async def monitor_start(self):
        if self.policy == "none":
            return
        # elif self.policy == "TTFT_threshold":
        #     self.monitor_tasks.append(
        #         asyncio.create_task(self.monitor_TTFT_threshold()))
        # elif self.policy == "memory_threshold":
        #     self.monitor_tasks.append(
        #         asyncio.create_task(self.monitor_memory_threshold()))
        elif self.policy == "reactive":
            self.monitor_tasks.append(
                asyncio.create_task(self.monitor_memory_threshold()))
        # elif self.policy == "workload_monitor":
        #     self.monitor_tasks.append(
        #         asyncio.create_task(self.monitor_workload()))
        # elif self.policy == "workload_predictor":
        #     self.monitor_tasks.append(
        #         asyncio.create_task(self.monitor_workload_predictor()))
        elif self.policy == "proactive":
            self.monitor_tasks.append(
                asyncio.create_task(self.monitor_workload_predictor()))
        elif self.policy == "proactive_dry":
            # self.monitor_tasks.append(
            #     asyncio.create_task(self.monitor_workload_predictor_dry()))
            self.monitor_tasks.append(
                asyncio.create_task(self._sub_monitor_workload_predictor(self.interval)))
        elif self.policy == "hybrid":
            self.monitor_tasks.append(
                asyncio.create_task(self.multi_monitor_workload_predictor_memory_threshold()))
            # self.monitor_tasks.append(
            #     asyncio.create_task(self.monitor_workload_predictor_memory_threshold()))
        # elif self.policy == "scheduler_load":
        #     self.monitor_tasks.append(
        #         asyncio.create_task(self.monitor_scheduler_load()))
        elif self.policy == "scheduler_lookahead":
            self.monitor_tasks.append(
                asyncio.create_task(self.monitor_scheduler_lookahead()))
        elif self.policy == "scheduler_lookahead_dete":
            self.monitor_tasks.append(
                asyncio.create_task(self.monitor_scheduler_lookahead_dete()))
        elif self.policy == "scheduler_lookahead_dete_emergency":
            self.monitor_tasks.append(
                asyncio.create_task(self.monitor_scheduler_lookahead_dete_emergency()))
        elif self.policy == "preserve":
            # self.monitor_tasks.append(
            #     asyncio.create_task(self.monitor_workload_predictor_scheduler_lookahead()))  
            # self.monitor_tasks.append(
            #     asyncio.create_task(self.multi_monitor_workload_predictor_scheduler_lookahead()))        
            self.monitor_tasks.append(
                asyncio.create_task(self.multi_monitor_workload_predictor_scheduler_lookahead_emergency()))  
        else:
            raise ValueError(f"Unknown scaler policy: {self.policy}")
    
        logger.info(f"Scaler policy: {self.policy}, monitoring interval: {self.interval}s.")


    async def monitor_stop(self):
        if self.monitor_tasks:
            self.monitor_stop_event.set()
            await asyncio.gather(*self.monitor_tasks)
            logger.info("Scaler monitor stopped.")


    async def scale_to(self, num_instances):
        num_instances = int(num_instances)
        if num_instances < 2: 
            if self.num_instances == 1:
                return
            num_instances = 1
        if num_instances > self.num_instance_slots: 
            if self.num_instances == self.num_instance_slots:
                return
            num_instances = self.num_instance_slots

        # if locked, inmediately cancel the scaling
        if not self.scaling_lock.locked():
            async with self.scaling_lock:
                # Cold start
                logger.info(f"Scaling from {self.num_instances} to {num_instances} instances, would take {self.cold_start_time} secs.")
                await asyncio.sleep(self.cold_start_time)

                self.scheduler.scale_instances_to(num_instances)
                self.num_instances = num_instances
                logger.info(f"Scaled to {self.num_instances} instances.")
                
                ''' Freezed after sucessful scaling
                1. Prevent thrashing
                2. Prevent over-scale
                '''
                await asyncio.sleep(self.scale_freeze_time)
                # self.num_instances = num_instances
    
    # Do not care about the global scaler lock, concurrency safty handled by caller
    async def emergent_scale_up(self, num_instances):
        num_instances = int(num_instances)
        if num_instances < 2: 
            if self.num_instances == 1:
                return
            num_instances = 1
        if num_instances > self.num_instance_slots: 
            if self.num_instances == self.num_instance_slots:
                return
            num_instances = self.num_instance_slots
            
        logger.info(f"Emergent scaling from {self.num_instances} to {num_instances} instances, would take {self.cold_start_time} secs.")
        await asyncio.sleep(self.cold_start_time)

        self.scheduler.scale_instances_to(num_instances)
        self.num_instances = num_instances
        logger.info(f"Emergent scaled to {self.num_instances} instances.")
        
        # # Additional freeze time
        # freeze_time = self.scale_freeze_time
        # while self.scaling_lock.locked():
        #     await asyncio.sleep(1)
        #     freeze_time -= 1
        #     if freeze_time <= 0:
        #         break
        # async with self.scaling_lock:
        #     asyncio.create_task(self.lock_for(self.scaling_lock, freeze_time))
            # self.num_instances = num_instances
    

    async def lock_for(self, lock, time):
        async with lock:
            await asyncio.sleep(time)


    # Old style: Better not use
    async def monitor_TTFT_threshold(self):
        ttft_threshold_up = 1.5
        ttft_threshold_down = 0.15

        slots_ttft_acc = [0.0] * self.num_instance_slots
        slots_req_acc = [0.0] * self.num_instance_slots
        
        async def update_acc(slots_ttft_acc, slots_req_acc):
            instance_slots_ttft = []
            _slots_ttft_acc, _slots_req_acc = await self.get_instances_slots_TTFT_req_acc()
            for i in range(self.num_instance_slots):
                ''' None value: metrics fetching failed  
                    case-1: occassionally event, 
                    case-2: instance overloaded, use the last value as reference 
                            (could be almost overloaded, thus should increase the value)  
                '''
                if _slots_ttft_acc[i] is None or _slots_req_acc[i] is None:
                    _slots_ttft_acc[i] = slots_ttft_acc[i] * 1.2 if _slots_ttft_acc[i] is None else _slots_ttft_acc[i]
                    _slots_req_acc[i] = slots_req_acc[i] if _slots_req_acc[i] is None else _slots_req_acc[i]
                    instance_slots_ttft.append(self.instance_slots_ttft[i] * 1.2)

                if _slots_req_acc[i] - slots_req_acc[i] < 1:
                    instance_slots_ttft.append(0.0)
                else:
                    instance_slots_ttft.append(
                        (_slots_ttft_acc[i] - slots_ttft_acc[i]) / (_slots_req_acc[i] - slots_req_acc[i]))

            self.instance_slots_ttft = instance_slots_ttft
            return _slots_ttft_acc, _slots_req_acc


        # Don't scale during the first init time
        slots_ttft_acc, slots_req_acc = await update_acc(slots_ttft_acc, slots_req_acc)
        await asyncio.sleep(self.interval)


        # logger.info("Scaler monitor loop started.")
        while not self.monitor_stop_event.is_set():
            slots_ttft_acc, slots_req_acc = await update_acc(slots_ttft_acc, slots_req_acc)

            ttfts = self.instance_slots_ttft[:self.num_instances]
            ttft = sum(ttfts) / len(ttfts)
            logger.info("Scaler monitored average TTFT: %f", ttft)

            # scale up / down
            if ttft > ttft_threshold_up:
                asyncio.create_task(self.scale_to(self.num_instances + 1))
            elif ttft < ttft_threshold_down:
                asyncio.create_task(self.scale_to(self.num_instances - 1))

            await asyncio.sleep(self.interval)

        # logger.info("Scaler monitor loop stopped.")


    def get_instances_slots_ttft(self):
        return self.instance_slots_ttft
    
    def get_instances_ttft(self):
        return self.instance_slots_ttft[:self.num_instances]

    async def get_instances_slots_TTFT_acc(self):
        return [
            await instance.get_openai_metrics_TTFT_acc()
            for instance in self.instance_slots
        ]

    async def get_instances_slots_TTFT_req_acc(self):
        ttft_acc_req_acc = [
            await instance.get_openai_metrics_TTFT_req_acc()
            for instance in self.instance_slots
        ]
        ttft_acc = [item[0] for item in ttft_acc_req_acc]
        req_acc = [item[1] for item in ttft_acc_req_acc]
        return ttft_acc, req_acc


        # # v1. Scale depend only on the total tokens
        # if total_prediction > MIU_ALL_MAX:
        #     asyncio.create_task(self.scale_to(self.num_instances + 2))
        # elif total_prediction < MIU_ALL_MIN:
        #     asyncio.create_task(self.scale_to(self.num_instances - 2))

        # # v2. miu_p, miu_d, miu_a  -> diff
        # num_inst_diff_p = float(prompt_prediction_2) / MIU["Azure_code"]["miu_p"] * self.interval / MIU_INTERVAL - self.num_instances
        # num_inst_diff_d = float(response_prediction_2) / MIU["Azure_code"]["miu_d"] * self.interval / MIU_INTERVAL - self.num_instances
        # num_inst_diff_a = float(total_prediction_2) / MIU["Azure_code"]["miu_a"] * self.interval / MIU_INTERVAL - self.num_instances

        # num_inst_diff = max(num_inst_diff_p, num_inst_diff_d, num_inst_diff_a)

        # increase_threshold = 0.0
        # decrease_threshold = -1.2
        # if num_inst_diff > increase_threshold:
        #     asyncio.create_task(self.scale_to(self.num_instances + 1))
        # elif num_inst_diff < decrease_threshold:
        #     asyncio.create_task(self.scale_to(self.num_instances - 1))
        
        # if num_inst_diff_p > increase_threshold or \
        #     num_inst_diff_d > increase_threshold or \
        #     num_inst_diff_a > increase_threshold:
        #     asyncio.create_task(self.scale_to(self.num_instances + 1))
        # elif num_inst_diff_p < decrease_threshold and \
        #     num_inst_diff_d < decrease_threshold and \
        #     num_inst_diff_a < decrease_threshold:
        #     asyncio.create_task(self.scale_to(self.num_instances - 1))

        # if prompt_prediction_2 > MIU["Azure_code"]["miu_p"] or \
        #    response_prediction_2 > MIU["Azure_code"]["miu_d"] or \
        #    total_prediction_2 > MIU["Azure_code"]["miu_a"]:
        #     asyncio.create_task(self.scale_to(self.num_instances + 2))
        # elif prompt_prediction_2 < MIU["Azure_code"]["miu_p"] * decrease_factor and \
        #      response_prediction_2 < MIU["Azure_code"]["miu_d"] * decrease_factor and \
        #      total_prediction_2 < MIU["Azure_code"]["miu_a"] * decrease_factor:
        #     asyncio.create_task(self.scale_to(self.num_instances - 2))


    # Hardcode: 1 instance max capacity
    MIU = {
        "Azure_code_peak": {
            # P:D = 21:1
            "miu_p": 34000,
            "miu_d": 2150,
            "miu_a": 35000,
        },
        "Azure_conv_peak": {
            # P:D = 9.35:1
            "miu_p": 19000,
            "miu_d": 2050,
            "miu_a": 21000,
        } 
    }
    MIU_INTERVAL = 2.0

    def _policy_workload_predictor(self, num_instances, lock):
        def update_acc(prompt_acc, response_acc):
            _prompt_acc = self.scheduler.get_prompt_tokens_acc()
            _response_acc = self.scheduler.get_response_tokens_acc()
            prompt_prev = _prompt_acc - prompt_acc
            response_prev = _response_acc - response_acc
            return _prompt_acc, _response_acc, prompt_prev, response_prev 

        miu_p = self.MIU[self.workload_name]["miu_p"] * self.MIU_INTERVAL
        miu_d = self.MIU[self.workload_name]["miu_d"] * self.MIU_INTERVAL
        miu_a = self.MIU[self.workload_name]["miu_a"] * self.MIU_INTERVAL

        # groundtruth prev up-ending window (group agg by sum)
        self.vars["prompt_acc"], self.vars["response_acc"], prompt_prev, response_prev = \
            update_acc(self.vars["prompt_acc"], self.vars["response_acc"])
        total_prev = prompt_prev + response_prev
        self.prompt_predictor.add_history(prompt_prev)
        self.response_predictor.add_history(response_prev)

        # predict next window
        prompt_prediction_1 = self.prompt_predictor.predict_next_window()
        self.prompt_predictor.add_history(prompt_prediction_1)
        prompt_prediction_2 = self.prompt_predictor.predict_next_window()
        self.prompt_predictor.rollback_history(1)
        self.prompt_predictor.update_history(prompt_prev)
        
        response_prediction_1 = self.response_predictor.predict_next_window()
        self.response_predictor.add_history(response_prediction_1)
        response_prediction_2 = self.response_predictor.predict_next_window()
        self.response_predictor.rollback_history(1)
        self.response_predictor.update_history(response_prev)

        total_prediction_2 = prompt_prediction_2 + response_prediction_2
        logger.info(f"Scaler prev groundtruth: {int(total_prev)}, next prediction: {int(total_prediction_2)}")

        # # v3. miu_p, miu_d, miu_a  -> max_num
        num_inst_p = float(prompt_prediction_2) / miu_p
        num_inst_d = float(response_prediction_2) / miu_d
        num_inst_a = float(total_prediction_2) / miu_a
        # num_inst_p = float(prompt_prev) / miu_p
        # num_inst_d = float(response_prev) / miu_d
        # num_inst_a = float(prompt_prev + response_prev) / miu_a
        num_inst = max(num_inst_p, num_inst_d, num_inst_a)

        limiter = None
        if num_inst == num_inst_p:  limiter = "prompt  "
        if num_inst == num_inst_d:  limiter = "response"
        if num_inst == num_inst_a:  limiter = "total   "
        logger.info(f"Scaler #Instance limited by {limiter} (p={num_inst_p} d={num_inst_d} a={num_inst_a})")

        # global best seting
        PADDING_UP, PADDING_DOWN = 0.2, 0.3
        num_inst_up = np.ceil(num_inst + PADDING_UP)
        num_inst_down = np.ceil(num_inst + PADDING_DOWN)
        
        if lock.locked():
            return num_instances
        elif num_inst_up > num_instances:
            # asyncio.create_task(self.scale_to(int(num_inst_up)))
            asyncio.create_task(self.lock_for(lock, self.scale_freeze_time))
            logger.info(f"Wkld pred scaling up to {int(num_inst_up)}")
            return int(num_inst_up)
        elif num_inst_down < num_instances:
            # asyncio.create_task(self.scale_to(int(num_inst_down)))
            asyncio.create_task(self.lock_for(lock, self.scale_freeze_time))
            logger.info(f"Wkld pred scaling down to {int(num_inst_down)}")
            return int(num_inst_down)
        else:
            return num_instances

    async def monitor_workload_predictor(self):
        # Don't scale during the first init time
        self.vars["prompt_acc"] = self.scheduler.get_prompt_tokens_acc()
        self.vars["response_acc"] = self.scheduler.get_response_tokens_acc()
        await asyncio.sleep(self.interval)
        
        lock = asyncio.Lock()
        while not self.monitor_stop_event.is_set():
            num_instances = self._policy_workload_predictor(self.num_instances, lock)
            if self.num_instances != num_instances:
                asyncio.create_task(self.scale_to(num_instances))

            await asyncio.sleep(self.interval)

    # # Debugging mode: Only record and log the groundtruth workload 
    # async def monitor_workload_predictor_dry(self):
    #     # Don't scale during the first init time
    #     self.vars["prompt_acc"] = self.scheduler.get_prompt_tokens_acc()
    #     self.vars["response_acc"] = self.scheduler.get_response_tokens_acc()
    #     await asyncio.sleep(self.interval)

    #     lock = asyncio.Lock()
    #     while not self.monitor_stop_event.is_set():
    #         # # groundtruth prev up-ending window (group agg by sum)
    #         # prompt_acc, response_acc, prompt_prev, response_prev = \
    #         #     update_acc(prompt_acc, response_acc)
    #         # total_prev = prompt_prev + response_prev
    #         # logger.info(f"Scaler prev groundtruth: prompt={prompt_prev}, response={response_prev}, total={int(total_prev)}")
            
    #         _ = self._policy_workload_predictor(self.num_instances, lock)
            
    #         await asyncio.sleep(self.interval)
        

    def _policy_scheduler_load(self, num_instances, lock):
        EMA_ALPHA = 0.9
        EMA_THRESHOLD_UP = 70_000
        EMA_THRESHOLD_DOWN = 15_000
        
        def update_load_EMA(load_EMA):
            _load = self.scheduler.get_instances_avg_load()
            load_EMA = EMA_ALPHA * _load + (1 - EMA_ALPHA) * load_EMA
            return load_EMA
        
        self.vars["inst_avg_load_EMA"] = update_load_EMA(self.vars["inst_avg_load_EMA"])
        logger.info(f"Scaler monitored average load: {self.vars['inst_avg_load_EMA']}")
        
        # scale up / down
        if lock.locked():
            return num_instances
        elif self.vars["inst_avg_load_EMA"] > EMA_THRESHOLD_UP:
            asyncio.create_task(self.lock_for(lock, self.scale_freeze_time))
            return num_instances + 1
        elif self.vars["inst_avg_load_EMA"] < EMA_THRESHOLD_DOWN:
            asyncio.create_task(self.lock_for(lock, self.scale_freeze_time))
            return num_instances - 1
        else:
            return num_instances
        
    
    async def monitor_scheduler_load(self):
        # Don't scale during the first init time
        self.vars["inst_avg_load_EMA"] = self.scheduler.get_instances_avg_load() # Init t=0
        await asyncio.sleep(self.interval)

        lock = asyncio.Lock()
        while not self.monitor_stop_event.is_set():
            num_instances = self._policy_scheduler_load(self.num_instances, lock)
            if self.num_instances != num_instances:
                asyncio.create_task(self.scale_to(num_instances))
                
            await asyncio.sleep(self.interval)
            
            
    LOOKAHEAD_MAX_PROF = 16 * 2306  # max under offline profiling 
    LOOKAHEAD_THRESHOLD = {
        "Azure_code_peak": {
            "up": 0.7,
            "down": 0.2,
        },
        "Azure_conv_peak": {
            "up": 0.7,
            "down": 0.15,
        }
    }
    def _policy_scheduler_lookahead(self, num_instances, lock):
        lookahead_threshold_up = self.LOOKAHEAD_THRESHOLD[self.workload_name]["up"]
        lookahead_threshold_down = self.LOOKAHEAD_THRESHOLD[self.workload_name]["down"]
        avg_lookahead = self.scheduler.get_instances_avg_lookahead_max()
        avg_lookahead_ratio = avg_lookahead / self.LOOKAHEAD_MAX_PROF
        self.vars["avg_lookahead_ratio"] = avg_lookahead_ratio
        logger.info(f"Scaler monitored average lookahead: {avg_lookahead} ({avg_lookahead_ratio}), should in range [{lookahead_threshold_down}, {lookahead_threshold_up}]")
        
        # scale up / down
        if lock.locked():
            logger.info("Lookahead is locked.")
            return num_instances
        elif avg_lookahead_ratio > lookahead_threshold_up:
            logger.info(f"Lookahead scaling up.")
            asyncio.create_task(self.lock_for(lock, self.scale_freeze_time))
            if avg_lookahead_ratio > 0.85:
                return num_instances + 2
            else:
                return num_instances + 1
        elif avg_lookahead_ratio < lookahead_threshold_down:
            logger.info("Lookahead scaling down.")
            asyncio.create_task(self.lock_for(lock, self.scale_freeze_time))
            return num_instances - 1
        else:
            return num_instances
        
    
    LOOKAHEAD_HEALTHY = {
        "healthy": 0.4,
        "up": 0.7,
        "down": 0.2,
    }
    LOOKAHEAD_EMERGENCY = 0.8
    def _policy_scheduler_lookahead_dete(self, num_instances, lock):
        avg_lookahead = self.scheduler.get_instances_avg_lookahead_max()
        avg_lookahead_ratio = float(avg_lookahead) / self.LOOKAHEAD_MAX_PROF 
        self.vars["avg_lookahead_ratio"] = avg_lookahead_ratio
        
        num_inst_L = avg_lookahead_ratio * self.num_instances / self.LOOKAHEAD_HEALTHY["up"]
        num_inst_R = avg_lookahead_ratio * self.num_instances / self.LOOKAHEAD_HEALTHY["down"]
        num_inst_healthy = np.ceil(avg_lookahead_ratio * self.num_instances / self.LOOKAHEAD_HEALTHY["healthy"])
        if num_inst_healthy < num_instances - 2:  num_inst_healthy = num_instances - 2
        if num_inst_healthy > num_instances + 2:  num_inst_healthy = num_instances + 2
        
        logger.info(f"Scaler monitored average lookahead: {avg_lookahead} ({avg_lookahead_ratio}) -> num_inst: [{num_inst_L}, {num_inst_R}] {int(num_inst_healthy)}")
        
        if lock.locked():
            logger.info("Lookahead is locked.")
            return num_instances
        elif num_instances < num_inst_L:
            logger.info(f"Lookahead scaling up to {int(num_inst_healthy)}.")
            asyncio.create_task(self.lock_for(lock, self.scale_freeze_time))
            return num_inst_healthy
        elif num_instances > num_inst_R:
            logger.info(f"Lookahead scaling down to {int(num_inst_healthy)}.")
            asyncio.create_task(self.lock_for(lock, self.scale_freeze_time))
            return num_inst_healthy
        else:
            return num_instances
        
        
    async def monitor_scheduler_lookahead(self):
        self.vars["avg_lookahead_ratio"] = 0.0
        
        # Don't scale during the first init time
        await asyncio.sleep(self.interval)
        
        lock = asyncio.Lock()
        while not self.monitor_stop_event.is_set():
            num_instances = self._policy_scheduler_lookahead(self.num_instances, lock)
            if self.num_instances != num_instances:
                asyncio.create_task(self.scale_to(num_instances))
            await asyncio.sleep(self.interval)
            
    async def monitor_scheduler_lookahead_dete(self):
        self.vars["avg_lookahead_ratio"] = 0.0
        
        # Don't scale during the first init time
        await asyncio.sleep(self.interval)
        
        lock = asyncio.Lock()
        while not self.monitor_stop_event.is_set():
            num_instances = self._policy_scheduler_lookahead_dete(self.num_instances, lock)
            if self.num_instances != num_instances:
                asyncio.create_task(self.scale_to(num_instances))
            await asyncio.sleep(self.interval)
    
    async def monitor_scheduler_lookahead_dete_emergency(self):
        self.vars["avg_lookahead_ratio"] = 0.0
        
        # Don't scale during the first init time
        await asyncio.sleep(self.interval)
        
        emergency_lock = asyncio.Lock()
        lock = asyncio.Lock()
        while not self.monitor_stop_event.is_set():
            if self.vars["avg_lookahead_ratio"] > self.LOOKAHEAD_EMERGENCY and not emergency_lock.locked():
                num_inst_healthy = np.ceil(self.vars["avg_lookahead_ratio"] * self.num_instances / self.LOOKAHEAD_HEALTHY["healthy"])
                asyncio.create_task(self.emergent_scale_up(num_inst_healthy))
                asyncio.create_task(self.lock_for(emergency_lock, self.interval * 3))
            else:
                num_instances = self._policy_scheduler_lookahead_dete(self.num_instances, lock)
                if self.num_instances != num_instances:
                    asyncio.create_task(self.scale_to(num_instances))
            await asyncio.sleep(self.interval)
    
        
    MEM_THRESHOLD = {
        "Azure_code_peak": {
            "up": 0.92,
            "down": 0.1,
        },
        "Azure_conv_peak": {
            "up": 0.9,
            "down": 0.2,
        }
    }
    def _policy_memory_threshold(self, num_instances, lock):
        mem_threshold_up = self.MEM_THRESHOLD[self.workload_name]["up"]
        mem_threshold_down = self.MEM_THRESHOLD[self.workload_name]["down"]
        avg_mem = self.scheduler.get_instances_avg_mem()
        logger.info(f"Scaler monitored average memory: {avg_mem}")
            
        # scale up / down
        if lock.locked():
            return num_instances
        elif avg_mem > mem_threshold_up:
            # asyncio.create_task(self.scale_to(self.num_instances + 1))
            asyncio.create_task(self.lock_for(lock, self.scale_freeze_time))
            return num_instances + 1
        elif avg_mem < mem_threshold_down:
            # asyncio.create_task(self.scale_to(self.num_instances - 1))
            asyncio.create_task(self.lock_for(lock, self.scale_freeze_time))
            return num_instances - 1
        else:
            return num_instances
            
    async def monitor_memory_threshold(self):
        await asyncio.sleep(self.interval)
        lock = asyncio.Lock()
        while not self.monitor_stop_event.is_set():
            num_instances = self._policy_memory_threshold(self.num_instances, lock)
            if self.num_instances != num_instances:
                asyncio.create_task(self.scale_to(num_instances))
            
            await asyncio.sleep(self.interval)


    # async def monitor_workload_predictor_memory_threshold(self):
    #     relative_interval = 2  # division between 2 monitor interval
        
    #     # Don't scale during the first init time
    #     self.vars["prompt_acc"] = self.scheduler.get_prompt_tokens_acc()
    #     self.vars["response_acc"] = self.scheduler.get_response_tokens_acc()
    #     await asyncio.sleep(self.interval * relative_interval)
        
    #     relative_interval_i = 0
    #     num_instances_wkld = self.num_instances
    #     num_instances_mem = self.num_instances
    #     lock_1 = asyncio.Lock()
    #     lock_2 = asyncio.Lock()
    #     while not self.monitor_stop_event.is_set():
    #         if relative_interval_i % relative_interval == 0:
    #             num_instances_wkld = self._policy_workload_predictor(num_instances_wkld, lock_1)
    #         num_instances_mem = self._policy_memory_threshold(num_instances_mem, lock_2)
    #         num_instances = max(num_instances_wkld, num_instances_mem)
            
    #         if self.num_instances != num_instances:
    #             asyncio.create_task(self.scale_to(num_instances))

    #         relative_interval_i += 1
    #         await asyncio.sleep(self.interval)
        
        
        
    async def _sub_monitor_workload_predictor(self, interval):
        self.vars["num_inst_wkld"] = self.num_instances
        lock = asyncio.Lock()
        self.vars["prompt_acc"] = self.scheduler.get_prompt_tokens_acc()
        self.vars["response_acc"] = self.scheduler.get_response_tokens_acc()

        await asyncio.sleep(interval)
        while not self.monitor_stop_event.is_set():
            self.vars["num_inst_wkld"] = self._policy_workload_predictor(self.vars["num_inst_wkld"], lock)

            await asyncio.sleep(interval)
            
    async def _sub_monitor_memory_threshold(self, interval):
        self.vars["num_inst_mem"] = self.num_instances
        lock = asyncio.Lock()
        
        await asyncio.sleep(interval)
        while not self.monitor_stop_event.is_set():
            self.vars["num_inst_mem"] = self._policy_memory_threshold(self.vars["num_inst_mem"], lock)
            await asyncio.sleep(interval)
            
    async def multi_monitor_workload_predictor_memory_threshold(self):
        relative_interval = 2  # division between 2 monitor interval
        asyncio.create_task(self._sub_monitor_workload_predictor(self.interval * relative_interval))
        asyncio.create_task(self._sub_monitor_memory_threshold(self.interval))
        
        await asyncio.sleep(self.interval * relative_interval)
        assert "num_inst_wkld" in self.vars and "num_inst_mem" in self.vars
        
        while not self.monitor_stop_event.is_set():
            num_instances = float(self.vars["num_inst_wkld"] + self.vars["num_inst_mem"]) / 2
            num_instances = np.ceil(num_instances)
            if self.num_instances != num_instances:
                asyncio.create_task(self.scale_to(num_instances))  # Lock by global scaling lock
            await asyncio.sleep(self.interval)
    
    
    
    # async def monitor_workload_predictor_scheduler_lookahead(self):
    #     relative_interval = 4 # !!!!!!!!!!!
    #     # Don't scale during the first init time
    #     self.vars["prompt_acc"] = self.scheduler.get_prompt_tokens_acc()
    #     self.vars["response_acc"] = self.scheduler.get_response_tokens_acc()
    #     await asyncio.sleep(self.interval * relative_interval)
        
    #     relative_interval_i = 0
    #     lock_1 = asyncio.Lock()
    #     lock_2 = asyncio.Lock()
    #     num_instances_wkld = self.num_instances
    #     num_instances_lookahead = self.num_instances
    #     while not self.monitor_stop_event.is_set():
    #         if relative_interval_i % relative_interval == 0:            
    #             num_instances_wkld = self._policy_workload_predictor(num_instances_wkld, lock_1)
    #         num_instances_lookahead = self._policy_scheduler_lookahead(num_instances_lookahead, lock_2)
    #         num_instances = max(num_instances_wkld, num_instances_lookahead)
            
    #         if self.num_instances != num_instances:
    #             asyncio.create_task(self.scale_to(num_instances))

    #         relative_interval_i += 1
    #         await asyncio.sleep(self.interval)
    
    
    
    async def _sub_monitor_scheduler_lookahead(self, interval):
        self.vars["num_inst_lookahead"] = self.num_instances
        self.vars["avg_lookahead_ratio"] = 0.0
        lock = asyncio.Lock()
        
        await asyncio.sleep(interval)
        while not self.monitor_stop_event.is_set():
            self.vars["num_inst_lookahead"] = self._policy_scheduler_lookahead(
                self.vars["num_inst_lookahead"], lock)
            await asyncio.sleep(interval)
            
    async def multi_monitor_workload_predictor_scheduler_lookahead(self):
        relative_interval = 4 # !!!!!!!!!!!
        asyncio.create_task(self._sub_monitor_workload_predictor(self.interval * relative_interval))
        asyncio.create_task(self._sub_monitor_scheduler_lookahead(self.interval))
        
        await asyncio.sleep(self.interval * relative_interval)
        assert "num_inst_wkld" in self.vars and "num_inst_lookahead" in self.vars
        
        while not self.monitor_stop_event.is_set():
            num_instances = np.ceil(max(self.vars["num_inst_wkld"], self.vars["num_inst_lookahead"]))
            if self.num_instances != num_instances:
                asyncio.create_task(self.scale_to(num_instances))  # Lock by global scaling lock
            await asyncio.sleep(self.interval)
            
            
    async def multi_monitor_workload_predictor_scheduler_lookahead_emergency(self):
        relative_interval = 4 # !!!!!!!!!!!
        asyncio.create_task(self._sub_monitor_workload_predictor(self.interval * relative_interval))
        asyncio.create_task(self._sub_monitor_scheduler_lookahead(self.interval))
        
        await asyncio.sleep(self.interval * relative_interval)
        assert "num_inst_wkld" in self.vars and "num_inst_lookahead" in self.vars and "avg_lookahead_ratio" in self.vars
        
        emergency_lock = asyncio.Lock()
    
        while not self.monitor_stop_event.is_set():
            num_instances = np.ceil(max(self.vars["num_inst_wkld"], self.vars["num_inst_lookahead"]))
            logger.info(f"2 Sub monitors: wkld={self.vars['num_inst_wkld']} lookahead={self.vars['num_inst_lookahead']} ({self.num_instances} -> {num_instances})") 
                        
            if self.vars["avg_lookahead_ratio"] > self.LOOKAHEAD_EMERGENCY and not emergency_lock.locked():
                num_inst_healthy = np.floor(self.vars["avg_lookahead_ratio"] * self.num_instances / self.LOOKAHEAD_HEALTHY["healthy"])
                self.vars["num_inst_lookahead"] = num_inst_healthy
                
                logger.info(f"Emergency lookahead scaling up to {num_inst_healthy}.")
                asyncio.create_task(self.emergent_scale_up(num_inst_healthy))
                asyncio.create_task(self.lock_for(emergency_lock, self.interval * 3))  # Quick Lock by caller
            
            elif self.num_instances != num_instances:
                logger.info(f"Normal lookahead scaling to {num_instances}.")
                asyncio.create_task(self.scale_to(num_instances))  # Lock by global scaling lock
            
            await asyncio.sleep(self.interval)
            
        

    '''
    df_train.columns: ['Timestamp', 'Request tokens', 'Response tokens']
    '''
    def train_workload_predictor(self, df_train):
        logger.info(f"Training workload predictor with {len(df_train)} requests trace data.")

        time_window = self.interval
        df_train['Timewindow'] = (df_train['Timestamp'] // time_window) * time_window
        df_grouped = df_train.groupby('Timewindow').agg({
            'Request tokens': 'sum',
            'Response tokens': 'sum'
        }).reset_index()[1:-1]

        df_prompt = df_grouped[['Request tokens']]
        df_prompt.columns = ['y']
        self.prompt_predictor = WorkloadPredictor(df_prompt)
        logger.info("Prompt predictor trained.")

        df_response = df_grouped[['Response tokens']]
        df_response.columns = ['y']
        self.response_predictor = WorkloadPredictor(df_response)
        logger.info("Response predictor trained.")

        self.df_train = df_train

    '''
    df_test.columns: ['Timestamp', 'Request tokens', 'Response tokens']
    '''
    def test_workload_predictor(self, df_test):
        time_window = self.interval
        df_test['Timewindow'] = (df_test['Timestamp'] // time_window) * time_window
        df_grouped = df_test.groupby('Timewindow').agg({
            'Request tokens': 'sum',
            'Response tokens': 'sum'
        }).reset_index()[1:-1]

        df_prompt = df_grouped[['Request tokens']]
        df_prompt.columns = ['y']
        prompt_y_test_mean = df_prompt['y'].mean()
        predictions = []
        for y in df_prompt['y']:
            next_prediction = self.prompt_predictor.predict_next_window()
            predictions.append(next_prediction)
            self.prompt_predictor.add_history(y)
        prompt_mae = mean_absolute_error(df_prompt['y'], predictions)
        logger.info(f"Prompt MAE: {prompt_mae}")
        logger.info(f"Prompt MAE percentage: {prompt_mae / prompt_y_test_mean}") 

        df_response = df_grouped[['Response tokens']]
        df_response.columns = ['y']
        response_y_test_mean = df_response['y'].mean()
        predictions = []
        for y in df_response['y']:
            next_prediction = self.response_predictor.predict_next_window()
            predictions.append(next_prediction)
            self.response_predictor.add_history(y)
        response_mae = mean_absolute_error(df_response['y'], predictions)
        logger.info(f"Response MAE: {response_mae}")
        logger.info(f"Response MAE percentage: {response_mae / response_y_test_mean}")

        # Test data added to history, should re-train the predictor based on df_train only 
        self.train_workload_predictor(self.df_train)
    
