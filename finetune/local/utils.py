import sys
import os
import glob
import random 
import numpy as np
import json
import re
import logging
import argparse
import librosa


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

def generate_random_string():
    # 获取当前进程ID
    pid = os.getpid()
    # 获取当前时间戳（精确到毫秒）
    timestamp = get_beijing_timestamp()
    # 生成随机数（0-9999）
    rand_num = random.randint(0, 9999)
    # 组合成唯一字符串
    unique_str = f"{timestamp}_{pid}_{rand_num}"
    return unique_str



def get_logger_file(log_file,  std_out=False):
    # 创建 logger 实例
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # 设置日志级别
    # 移除所有默认处理器（避免控制台输出）
    logger.handlers.clear()
    logger.propagate = False

    # 创建文件处理器（追加模式）
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # 文件日志级别
    formatter = logging.Formatter('[%(asctime)s %(processName)s:%(process)d %(name)s %(filename)s:%(lineno)d %(funcName)s %(levelname)s]: %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    if std_out:
        # 创建控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)  # 控制台日志级别
        formatter = logging.Formatter('[%(asctime)s %(name)s %(levelname)s]: %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger



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

def read_wav_scp_file(scp_file):
    utts = []
    wav2scp = {}
    with open(scp_file, 'r', encoding='utf-8') as f:
        for line in f:
            utt, wav_path = line.strip().split()
            utts.append(utt)
            wav2scp[utt] = wav_path
    return utts, wav2scp

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




def get_parse():
    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument('--wav_in', type=str, 
                        default=None,
                        help='wav file')

    parser.add_argument('--wav_scp', type=str, 
                        default=None,
                        help='wav_scp file')
    parser.add_argument('--wav_dir', type=str, 
                        default=None,
                        help='Path to the wav directory')
    
    parser.add_argument('--out_dir', type=str, 
                        default=None,
                        help='Path to the output directory')
    

    return parser


def get_wavs_from_args(args):
    
    datas = []
    if args.wav_in:
        utt = os.path.basename(args.wav_in)[:-4]
        datas.append([utt, args.wav_in])
    elif args.wav_scp:
        utts, wav2scp = read_wav_scp_file(args.wav_scp)
        for utt in utts:
            datas.append([utt, wav2scp[utt]])
    elif args.wav_dir:
        wavs = get_dir_files(args.wav_dir, '.wav')
        for wav_file in wavs:
            utt = os.path.basename(wav_file)[:-4]
            datas.append([utt, wav_file])
    else:
        print(f'error: parse args error: {args}')
    
    return datas



def draw_spec(ax, speech, sample_rate):
    n_fft = 1024
    hop_length=512
    n_mels=120
    mel_spec = librosa.feature.melspectrogram(y=speech, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalize data
    min_level_db=-100
    mel_spec_db = (log_mel_spec - min_level_db) / (-min_level_db)

    # 显示 Mel 频谱图
    librosa.display.specshow(mel_spec_db, ax=ax, x_axis='time', y_axis='mel', sr=sample_rate, hop_length=hop_length,fmax=sample_rate/2)

    # 添加颜色条
    # plt.colorbar(format="%+2.0f dB")