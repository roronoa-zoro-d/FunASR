import sys
import os
import glob
import random 
import numpy as np
import json
import re

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


# 数据集划分
def split_list_randomly(input_list, ratios):
    """
    将输入列表按指定比例随机划分为多个子列表。
    
    :param input_list: 要划分的列表
    :param ratios: 一个包含比例的列表或元组，例如 [0.7, 0.2, 0.1]
    :return: 包含子列表的列表
    """
    # 首先对列表进行深拷贝并打乱顺序
    shuffled_list = input_list.copy()
    random.shuffle(shuffled_list)
    
    # 计算每个子列表的大小
    total_length = len(shuffled_list)
    split_indices = []
    cumulative_sum = 0
    
    for ratio in ratios[:-1]:  # 不包括最后一个比例，因为它是剩余的部分
        cumulative_sum += ratio
        split_indices.append(int(cumulative_sum * total_length))
    
    # 根据计算出的索引划分列表
    split_lists = []
    start_index = 0
    for index in split_indices:
        split_lists.append(shuffled_list[start_index:index])
        start_index = index
    # 添加最后一个子列表
    split_lists.append(shuffled_list[start_index:])
    
    return split_lists


# 读写jsonl数据
def write_jsonl_data(datas, out_file):
    with open(out_file, 'w', encoding='utf-8') as f:
        for data in datas:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

def read_jsonl_data(in_file):
    datas = []
    with open(in_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            datas.append(data)
    return datas

def read_json_data(in_file):
    with open(in_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def get_dir_files(in_dir, suffix, recursive=True):
    if recursive:
        wav_files = glob.glob(os.path.join(in_dir, '**', f'*{suffix}'), recursive=recursive)
    else:
        wav_files = glob.glob(os.path.join(in_dir,  f'*{suffix}'), recursive=recursive)
    return wav_files


def get_wav_scp(in_dir, suffix, recursive=True):
    wav_files = get_dir_files(in_dir, suffix, recursive)
    wav_scp = {}
    for wav_file in wav_files:
        utt_id = os.path.basename(wav_file).replace(suffix, '')
        wav_scp[utt_id] = wav_file
    return wav_scp

def remove_punctuation(text):
    # 匹配所有中文标点、英文标点、数字间的符号（保留字母数字和空格）
    pattern = r'[^\w\d]'  # 匹配非字母数字、的字符
    # 如果需要保留数字，可以改为：r'[^\w\s\d]'（但通常不需要）
    return re.sub(pattern, '', text)


def read_wav_scp(scp_file):
    utts = []
    wav2scp = {}
    
    with open(scp_file, 'r', encoding='utf-8') as f:
        for line in f:
            utt, wav_path = line.strip().split()
            utts.append(utt)
            wav2scp[utt] = wav_path
    
    return utts, wav2scp




def get_models(in_dir):
    models = glob.glob(os.path.join(in_dir,  f'model.pt*'), recursive=False)
    models_sorted = sorted(models, key=lambda x: os.path.getmtime(x), reverse=True)
    return models_sorted



def read_utts(filename):
    utts = []
    with open(filename, 'r') as f:
        for line in f:
            utt = line.strip()
            utts.append(utt)
    return utts


def read_text(filename):
    utts = []
    utt2text = {}
    
    with open(filename, 'r') as f:
        for line in f:
            buff = line.strip().split("\t")
            utt = buff[0]
            text = ""
            if len(buff) == 2:
                text = buff[1]
            utt2text[utt] = text
            utts.append(utt)
    
    return utts, utt2text