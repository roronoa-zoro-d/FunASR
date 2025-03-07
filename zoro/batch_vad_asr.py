import sys
import os
from funasr import AutoModel
import soundfile
import json

import multiprocessing
from multiprocessing import Pool

# import ray

"""
使用vad对客服通话的首句和尾句进行切分， 识别内容， 并保存， 
用于后续找到 首句和尾句识别不准的片段，进行声学微调
"""


# 获取时间戳
def get_beijing_timestamp():
    import ntplib, pytz
    from datetime import datetime

    # 创建NTP客户端
    client = ntplib.NTPClient()
    # 请求NTP服务器获取时间
    response = client.request("pool.ntp.org", version=3)
    # 将NTP时间转换为UTC时间
    ntp_time_utc = datetime.fromtimestamp(response.tx_time, pytz.utc)
    # 将UTC时间转换为北京时间（CST）
    beijing_tz = pytz.timezone('Asia/Shanghai')
    beijing_time = ntp_time_utc.astimezone(beijing_tz)
    # 格式化时间为年月日时
    timestamp = beijing_time.strftime("%Y%m%d%H")
    return timestamp



def read_wav_scp(wav_scp_file):
    utts = []
    wav2path = {}
    with open(wav_scp_file, 'r') as f:
        for line in f:
            utt, path = line.strip().split()
            utts.append(utt)
            wav2path[utt] = path
    print(f'read {len(utts)} utts')
    return utts, wav2path

def split_chunk(datas, num_chunk):
    chunks = []
    for st in range(num_chunk):
        chunk = [datas[i] for i in range(st, len(datas), num_chunk)]
        chunks.append(chunk)
    
    return chunks



def process_vad_asr(wav_datas, log_file, idx):
    
    gpu_id = idx % 3 + 1    # 不使用0卡
    device = f'cuda:{gpu_id}'
    vad_model = AutoModel(model="fsmn-vad", model_revision="v2.0.4", device=device)
    asr_model = AutoModel(model="paraformer-zh", model_revision="v2.0.4", device=device)
    
    
    writer = open(log_file, 'w')

    datas = []
    for i, [utt, wav_file] in enumerate(wav_datas):
        speech, sample_rate = soundfile.read(wav_file)
        num_point_ms = int(sample_rate/1000)
        
        speech_dur = len(speech) / sample_rate
        if speech_dur < 10:
            continue
        
        res_vad = vad_model.generate(input=wav_file, disable_pbar=True)
        segments = res_vad[0]['value']
        if len(segments) < 2:
            continue
        first_chunk = speech[num_point_ms*segments[0][0]:num_point_ms*segments[0][1]]
        final_chunk = speech[num_point_ms*segments[-1][0]:num_point_ms*segments[-1][1]]
        first_chunk_asr_res = asr_model.generate(input=first_chunk, disable_pbar=True)
        first_chunk_asr = first_chunk_asr_res[0]['text']
        final_chunk_asr_res = asr_model.generate(input=final_chunk, disable_pbar=True)
        final_chunk_asr = final_chunk_asr_res[0]['text']

        data = {
            "utt": utt,
            "first_chunk": {
                "vad_seg": segments[0],
                "asr_txt": first_chunk_asr,
            },
            "final_chunk": {
                "vad_seg": segments[-1],
                "asr_txt": final_chunk_asr,
            },
        }
        datas.append(data)
        writer.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        # if i >= 10:
        #     break
        
        if i % 100 == 0:
            print(f'process {idx} : processed {i} samples')
        
        
    
    writer.close()
    
    return datas
    

wav_scp_file = '/data/nas/dataset/asr/kefu/shidian/wav.scp'

num_worker = 24


utts, wav2path = read_wav_scp(wav_scp_file)
timestamp = get_beijing_timestamp()


log_files = [f'logs/res_asr_{timestamp}_{i}.jsonl' for i in range(num_worker)]
os.makedirs('logs', exist_ok=True)
split_utts = split_chunk(utts, num_worker)
split_datas = []
for chunk_utts in split_utts:
    chunk = [[utt, wav2path[utt]] for utt in chunk_utts]
    split_datas.append(chunk)
    


args = [(split_datas[i], log_files[i], i) for i in range(num_worker)]

# 使用进程池来处理音频文件
with Pool(processes=num_worker) as pool:
        # 使用map方法将音频文件列表分配给进程池中的工作进程
        results = pool.starmap(process_vad_asr, args)
        
        # for result in results:
        #     if result:  # 假设process_vad_asr可能返回一些信息
        #         print(result)



# 使用ray框架处理多并发