import random
import json
import sys
import os
import glob
from datetime import datetime
import re

"""
将一个文件夹下的所有音频和文本，划分成 train dev test 数据集，并生成对应的wav.scp text
"""

split_ratio = {
    'train': 0.8,
    'dev': 0.05,
    'test': 0.15
}


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



def get_files(tts_wav_dir, suffix):
    wav_files = glob.glob(os.path.join(tts_wav_dir, '**', f'*{suffix}'), recursive=True)
    
    print(f'find {suffix} :  {len(wav_files)} files')
    
    utt2wav = {}
    for wav_file in wav_files:
        utt_id = os.path.basename(wav_file).replace(suffix, '')
        utt2wav[utt_id] = wav_file
    
    return utt2wav



if __name__ == '__main__':
    in_wav_dir = sys.argv[1]
    out_data_dir = sys.argv[2]
    
    
    utt2wav = get_files(in_wav_dir, '.wav')
    utt2txt = get_files(in_wav_dir, '.txt')
    
    utts = utt2wav.keys() & utt2txt.keys()
    
    punctuation_pattern = r'[^\w\s\u4e00-\u9fff]'  # 添加了对中文字符的考虑
    datas = []
    for utt_id in utts:
        wav_file = utt2wav[utt_id]
        txt_file = utt2txt[utt_id]
        with open(txt_file, 'r', encoding='utf-8') as f:
            txt = f.read().strip()
            txt = re.sub(punctuation_pattern, '', txt)
        datas.append({
            "utt": utt_id,
            "txt": txt,
            "wav": wav_file,
        })
    
    print(f'final get {len(datas)} wav-text' )
    
    num_data = len(datas)
    timestamp = get_beijing_timestamp()

    out_file = f"{out_data_dir}/data_tts_{timestamp}_{num_data}.jsonl"
    write_jsonl_data(datas, out_file)
    
    
    names = list(split_ratio.keys())
    ratios = [split_ratio[k] for k in names]
    split_lists = split_list_randomly(datas, ratios)

    for i, sublist in enumerate(split_lists):
        print(f"Sublist {i} length: {len(sublist)}")


    for i, name in enumerate(names):
        os.makedirs(os.path.join(out_data_dir, name), exist_ok=True)
        with open(f'{out_data_dir}/{name}/wav.scp', 'w') as f1, open(f'{out_data_dir}/{name}/text', 'w') as f2:
            for data in split_lists[i]:
                f1.write(f"{data['utt']} {data['wav']}\n")
                f2.write(f"{data['utt']} {data['txt']}\n")