import sys
import torch
import logging
import soundfile as sf 
import json


import ray
from ray import serve
from ray.serve.handle import DeploymentHandle


from starlette.requests import Request


from vad_engine import VAD_fsmn as VAD_model

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s-%(name)s-%(filename)s-%(funcName)s-%(lineno)d-%(levelname)s-%(process)d] %(message)s')

logger = logging.getLogger(__name__)

# 添加 StreamHandler 输出到终端
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter('[%(asctime)s-%(name)s-%(filename)s-%(funcName)s-%(lineno)d-%(levelname)s-%(process)d] %(message)s'))
logger.addHandler(stream_handler)

@serve.deployment(
    route_prefix="/vad/fsmn",
    num_replicas=3,
    ray_actor_options={'num_cpus':1, 'num_gpus':0.3, 
                       })
class VAD_Engine(object):
    def __init__(self) -> None:
        gpu_id = torch.cuda.current_device()
        device = 'cpu'
        if torch.cuda.is_available():
            device = f'cuda:{gpu_id}'
        
        self.vad_model = VAD_model(device=device)
    
    async def __call__(self, request: Request):
        task_data = await request.json()
        logger.info(f'################# server get task: {task_data}')
        wav_path = task_data['wav_path']
        
        speech_data, fs = sf.read(wav_path)
        
        return self.vad_model.vad_inference(speech_data,  fs=fs)
    

vad_app = VAD_Engine.bind()