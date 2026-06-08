from LLMServe.config import get_all_config
from LLMServe.global_scheduler import Scheduler
from LLMServe.logger import init_logger
import pandas as pd

logger = init_logger()

if __name__ == "__main__":
    config = get_all_config()

    global scheduler
    scheduler = Scheduler(config.scheduler_config, config.instance_config)
    df = pd.read_csv("../data/datasets/ShareGPT/cleaned.csv")
    for i in range(10):
        text = df.iloc[i]["prompts"]
        logger.info(f"Predicting: {scheduler.test_predictor(text)} Ground Truth: {df.iloc[i]['response_lens']}")