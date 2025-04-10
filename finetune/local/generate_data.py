import os
import glob
import sys
import re

# 生成 wav.scp text 文件


def get_files(tts_wav_dir, suffix):
    wav_files = glob.glob(os.path.join(tts_wav_dir, '**', f'*{suffix}'), recursive=True)
    
    print(f'find {suffix} :  {len(wav_files)} files')
    
    utt2wav = {}
    for wav_file in wav_files:
        utt_id = os.path.basename(wav_file).replace(suffix, '')
        utt2wav[utt_id] = wav_file
    
    return utt2wav


if __name__ == '__main__':
    
    in_dir = sys.argv[1]
    
    utt2wav = get_files(in_dir, '.wav')
    utt2txt_file = get_files(in_dir, '.txt')
    
    utts = utt2wav.keys() & utt2txt_file.keys()
    
    punctuation_pattern = r'[^\w\s\u4e00-\u9fff]'  # 添加了对中文字符的考虑
    utt2text = {}
    for utt in utts:
        wav_file = utt2wav[utt]
        txt_file = utt2txt_file[utt]
        with open(txt_file, 'r', encoding='utf-8') as f:
            txt = f.read().strip()
            txt = re.sub(punctuation_pattern, '', txt)
            utt2text[utt] = txt
    
    print(f'total {len(utts)} utts')

    
    with open(f'{in_dir}/wav.scp', 'w') as f1, open(f'{in_dir}/text', 'w') as f2:
        for utt in utts:    
            f1.write(f"{utt} {utt2wav[utt]}\n")
            f2.write(f"{utt} {utt2text[utt]}\n")