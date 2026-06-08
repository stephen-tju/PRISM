import asyncio
import os
import time
from tqdm.asyncio import tqdm
import pynvml

from LLMServe.config import get_all_config
from LLMServe.util import save_benchmark_result
from LLMServe.request_generater import Generator
from LLMServe.global_scheduler import Scheduler, Scaler
from LLMServe.logger import init_logger, setup_local_logger

logger = init_logger()



avail_openai_metrics = [
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_preemptions_total",
    "vllm:gpu_cache_usage_perc"
]

class BenchmarkMonitor:
    def __init__(self, scheduler, scaler):
        self.scheduler = scheduler
        self.scaler = scaler
        
        self.num_instance_slots = scheduler.get_num_instance_slots()
        self.monitor_task = None
        self.monitor_stop_event = asyncio.Event()
        # Ignore case: multiple GPUs per instance
        self.gpu_handles = [
            pynvml.nvmlDeviceGetHandleByIndex(gpu_id) 
            for gpu_id in range(self.num_instance_slots)
        ]
        self.gpu_utilizations = None
    
    def get_system_gpu_utilizations(self):
        return [
            pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handles[gpu_id]).gpu 
            for gpu_id in range(self.scaler.num_instances)
        ]

    async def monitor_start(self, profile_config, interval):
        self.monitor_task = asyncio.create_task(
            self.monitor_by_seconds(profile_config, interval))
        logger.info(f"Benchmark monitor started with {interval}s interval.")

    async def monitor_stop(self):
        if self.monitor_task:
            self.monitor_stop_event.set()
            await self.monitor_task
            logger.info("Benchmark monitor stopped.")


    async def monitor_by_seconds(self, profile_config, interval):
        monitor_logger = setup_local_logger(
            log_file_path=os.path.join(profile_config["case_path"], "monitor.log"),
            caller_file=__file__+":"+__name__,
        )
        info_id = 0
        # logger.info("Benchmark monitor loop started.")
        while not self.monitor_stop_event.is_set():
            await asyncio.sleep(interval)

            # Framework load metrics
            if info_id % 1 == 0:
                monitor_logger.debug(
                    "('Load', %s)", self.scheduler.get_instances_load_info()
                )
                # Lookahead test
                inst_0 = self.scheduler.instances[0]
                logger.debug(f"Inst-0 Lookahead index = {inst_0.lookahead.queue.head}, global current timestep = {inst_0.lookahead.global_current_timestep}")
            # vllm internal metrics
            if info_id % 1 == 0:
                try:
                    openai_metrics = await scheduler.get_instances_openai_metrics()
                    content_internal = [
                        { metric: instance_metrics[metric] for metric in avail_openai_metrics if metric in instance_metrics }
                        for instance_metrics in openai_metrics
                    ]
                    self.scheduler.update_gpu_memory_utilizations(
                        [dic["vllm:gpu_cache_usage_perc"] for dic in content_internal])
                    monitor_logger.debug("('Internal', %s)", content_internal)
                except Exception as e:
                    monitor_logger.error(f"Monitor failed to log Internal: {e}")
            # system external metrics
            if info_id % 1 == 0:
                try:
                    ttfts = scaler.get_instances_ttft()
                    gpu_utilizations = self.get_system_gpu_utilizations()
                    self.scheduler.update_gpu_utilizations(gpu_utilizations)
                    content_external = [
                        {
                            "ttft": float(ttfts[i]), "gpu_utilization": float(gpu_utilizations[i])
                        } for i in range(self.scaler.num_instances)
                    ]
                    monitor_logger.debug("('External', %s)", content_external)
                except Exception as e:
                    monitor_logger.error(f"Monitor failed to log External: {e}")
            info_id += 1
        # logger.info("Benchmark monitor loop stopped.")


async def send_request_at_timestamp(request_id, request, sleeptime):
    await asyncio.sleep(float(sleeptime))
    global scheduler
    st = time.time()

    result = await scheduler.handle_request(request_id, request)
    
    dur = time.time() - st
    result["frontend_latency"] = dur
    return result


async def benchmark(requests):
    results = []
    tasks = [
        asyncio.create_task(send_request_at_timestamp(request_id, request[:3], request[3] + 1))
        for request_id, request in enumerate(requests)
    ]
    await asyncio.sleep(1)
    for completed_task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        result = await completed_task
        results.append(result)

    return results


async def main():
    config = get_all_config()

    request_generator = Generator(
        request_config=config.request_config,
        model_name=config.instance_config["model_name"]
    )
    requests = request_generator.generate_requests()


    global scheduler
    scheduler = Scheduler(config.scheduler_config, config.instance_config)
    global scaler
    scaler = Scaler(scheduler, config.scheduler_config, config.request_config['workload'])
    if config.scheduler_config["scaler_policy"] != "reactive":
        scaler.train_workload_predictor(request_generator.get_workload_train_trace())
        # scaler.test_workload_predictor(request_generator.get_workload_test_trace())
    await scaler.monitor_start()

    global monitor
    monitor = BenchmarkMonitor(scheduler, scaler)
    await monitor.monitor_start(config.profile_config, interval=2)


    results = await benchmark(requests)


    await monitor.monitor_stop()
    await scaler.monitor_stop()

    logger.info("Total %d results finished.", len(results))
    save_benchmark_result(results, config)
    logger.info("Saved results to file.")
    
    # run_log_dir = os.path.join(os.path.dirname(__file__), "../results")
    # if os.path.exists(run_log_dir):
    #     os.system(f"{run_log_dir}/run.log* {case_path}")


if __name__ == "__main__":
    pynvml.nvmlInit()
    print("CUDA Driver Version:", pynvml.nvmlSystemGetDriverVersion())

    asyncio.run(main())