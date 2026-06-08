import torch
import os
from LLMServe.logger import init_logger
from .model import ResponsePredictor


logger = init_logger()


class LoadPredictor:
    def __init__(self, scheduler_config):
        self.model_path = scheduler_config['req_predictor_model_path']
        if not os.path.exists(self.model_path):
            logger.error(f"Model path {self.model_path} does not exist!")
            raise FileNotFoundError(f"Model path {self.model_path} does not exist!")        
        # self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = torch.device("cpu")
        self.model = ResponsePredictor().to(self.device)
        self.load_model()

    def load_model(self):
        load_mdoel = torch.load(self.model_path,
            weights_only=False,
            map_location=self.device,
        )
        self.model.bert.transformer.layer[-1] = load_mdoel['last_layer']
        self.model.cls = load_mdoel['cls']
        self.model.leanrable_prompts = load_mdoel['prompt']
    
    def predict(self, text):
        prediction = self.model([text], self.device).unsqueeze(0)
        return int(prediction.item())
    