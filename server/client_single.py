import asyncio
import aiohttp
import logging
import soundfile as sf
import torch
from collections import deque


import requests

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s-%(name)s-%(filename)s-%(funcName)s-%(lineno)d-%(levelname)s-%(process)d] %(message)s'
)
logger = logging.getLogger(__name__)


def read_wav_scp(filename):
    utts = []
    utt2wav = {}
    with open(filename, 'r' ) as f:
        for line in f:
            line = line.strip()
            if line:
                utt, wav_path = line.split()
                utt2wav[utt] = wav_path
                utts.append(utt)
    
    return utts, utt2wav



def get_vad(wav_path, url='http://127.0.0.1:8000/vad/fsmn/'):
    task = {"wav_path":wav_path, }
    result = {}
    try:
        # 发送POST请求
        response = requests.post(url, json=task)
        
        # 检查请求是否成功
        if response.status_code == 200:
            result = response.json()  # 假设服务器返回JSON格式的数据
        else:
            print(f"Failed to get a successful response for {utt}, status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        # 处理可能发生的异常，例如网络问题
        print(f"Request failed for {utt}: {e}")
    
    return result


def get_asr(wav_path, segs=[],  url='http://127.0.0.1:8000/asr/paraformer/'):
    task = {"wav_path":wav_path, "segs":segs}
    result = None
    try:
        # 发送POST请求
        response = requests.post(url, json=task)
        
        # 检查请求是否成功
        if response.status_code == 200:
            result = response.json()  # 假设服务器返回JSON格式的数据
        else:
            print(f"Failed to get a successful response for {utt}, status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        # 处理可能发生的异常，例如网络问题
        print(f"Request failed for {utt}: {e}")
    
    return result



wav_scp_file = '/data/nas/dataset/asr/kefu/shidian/wav1.scp'
wav_scp_file = '/data/nas/dataset/asr/kefu/huaian/wav.scp'


utts, utt2wav = read_wav_scp(wav_scp_file)


for i, utt in enumerate(utts):
    wav_path = utt2wav[utt]

    
    task = {"wav_path":wav_path, }
    
    print(f'utt: {utt} , wav_path: {wav_path}')
    
    vad_res = get_vad(wav_path)
    if vad_res is None:
        print(f'{utt} vad failed')
        break
    segs = vad_res['vad_segs']
    print(f'vad_res: {vad_res}')
    asr_res = get_asr(wav_path, segs=segs, url='http://127.0.0.1:8000/asr/paraformer/')
    print(f'asr_res: {asr_res}')
    
    asr_res = get_asr(wav_path, segs=segs, url='http://127.0.0.1:8000/asr/whisper_large/')
    print(f'asr_res: {asr_res}')

    asr_res = get_asr(wav_path, segs=segs, url='http://127.0.0.1:8000/asr/sense_voice_small/')
    print(f'sense_voice : asr_res: {asr_res}')

    asr_res = get_asr(wav_path, segs=segs, url='http://127.0.0.1:8001/asr/fireredasr/')
    print(f'fireredasr: asr_res: {asr_res}')

    break







