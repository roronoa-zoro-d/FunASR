import asyncio
import aiohttp
import logging
import multiprocessing
from itertools import islice
from functools import partial
import soundfile as sf
import json
import time

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s-%(name)s-%(filename)s-%(funcName)s-%(lineno)d-%(levelname)s-%(process)d] %(message)s'
)
logger = logging.getLogger(__name__)

VAD_URL = 'http://127.0.0.1:8000/vad/fsmn/'
ASR_URLS = [
    'http://127.0.0.1:8000/asr/paraformer/',
    # 'http://127.0.0.1:8000/asr/whisper_large/',
    'http://127.0.0.1:8000/asr/sense_voice_small/',
    'http://127.0.0.1:8001/asr/fireredasr/'
]

def read_wav_scp(filename):
    utts = []
    utt2wav = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                utt, wav_path = line.split()
                utt2wav[utt] = wav_path
                utts.append(utt)
    return utts, utt2wav

output_dir = '/data/nas/zhangjiayuan/temp/annos/kefu/'
wav_scp_file = '/data/nas/dataset/asr/kefu/huaian/wav.scp'
utts, utt2wav = read_wav_scp(wav_scp_file)

utts = utts[:8]

async def async_get_vad(session, wav_path):
    start_time = time.time()
    try:
        async with session.post(VAD_URL, json={"wav_path": wav_path}) as resp:
            if resp.status == 200:
                return await resp.json(), time.time() - start_time
            logger.error(f"VAD failed: {wav_path} status={resp.status}")
            return None, time.time() - start_time
    except Exception as e:
        logger.error(f"VAD error: {wav_path} {str(e)}")
        return None, time.time() - start_time

async def async_get_asr(session, wav_path, segs, url):
    start_time = time.time()
    try:
        async with session.post(url, json={"wav_path": wav_path, "segs": segs}) as resp:
            if resp.status == 200:
                return await resp.json(), time.time() - start_time
            logger.error(f"ASR failed: {url} {wav_path} status={resp.status}")
            return None, time.time() - start_time
    except Exception as e:
        logger.error(f"ASR error: {url} {wav_path} {str(e)}")
        return None, time.time() - start_time

async def process_one(session, utt):
    
    wav_path = utt2wav[utt]
    
    speech_data, fs = sf.read(wav_path)
    
    result = {}
    result['utt'] = utt
    result['wav_path'] = wav_path
    result['fs'] = fs
    result['dur'] = len(speech_data) / fs
    result['annos'] = []  # 包含多个切分方式下的结果,但目前只有一个结果anno
    
    stats = {}
    stats['speech_dur'] = len(speech_data) / fs
    stats['time_counter'] = []
    
    # 处理VAD
    vad_result, vad_use_time = await async_get_vad(session, wav_path)
    rtf_vad = vad_use_time / (len(speech_data) / fs)
    stats['time_counter'].append({"model_name": "fsmn_vad", "use_time": vad_use_time, "rtf": rtf_vad})
    if not vad_result:
        return result, stats
    

    anno = vad_result
    
    # 并发ASR请求
    segs = vad_result.get('vad_segs', [])
    tasks = [async_get_asr(session, wav_path, segs, url) for url in ASR_URLS]
    async_results = await asyncio.gather(*tasks)
    
  
    asr_results = []
    for asr_result,  asr_use_time in async_results:
        if asr_result:
            asr_results.append(asr_result)
            
            model_name = asr_result['model_name'].split('/')[-1]
            rtf = asr_use_time / result['dur']
            asr_stat = {
                "model_name": model_name,
                "use_time": asr_use_time,
                "rtf": rtf,
            }
            stats['time_counter'].append(asr_stat)

            
    
    
    anno['vad_asr_res'] = asr_results
    result['annos'].append(anno)
    
    # print(f'all_results: {result}')
    
    with open(f'{output_dir}/{utt}.json', 'w') as f:
        json.dump(result, f, ensure_ascii=False)
    
    
    return result, stats

async def process_chunk(utt_chunk, concurrency=10):
    semaphore = asyncio.Semaphore(concurrency)
    
    async def limited_task(session, path):
        async with semaphore:
            return await process_one(session, path)
    
    async with aiohttp.ClientSession() as session:
        tasks = [limited_task(session, utt) for utt in utt_chunk]
        return await asyncio.gather(*tasks)

def run_async_chunk(utt_chunk):
    return asyncio.run(process_chunk(utt_chunk))


def analysis_time(stats):
    
    res = {}
    model_names = set()
    for stat in stats:
        speech_dur = stat['speech_dur']
        res.setdefault('speech_dur', 0)
        res['speech_dur'] += speech_dur 
        for data in stat['time_counter']:
            model_name = data['model_name']
            use_time = data['use_time']
            rtf = data['rtf']
        
            res.setdefault(model_name, 0)
            res[model_name] += use_time
            model_names.add(model_name)
    
    num_utt = len(stats)
    total_speech_dur = res['speech_dur']
    for model_name in model_names:
        use_time = res[model_name]
        rtf = use_time / total_speech_dur
        print(f'process {num_utt} utt, total {total_speech_dur}s speech, model {model_name},  use {use_time:.2f}s, rtf: {rtf:.2f}')
    
    
    
    
            

def main():

    # 配置并行参数
    num_cpus = multiprocessing.cpu_count()
    num_cpus = 4
    chunk_size = len(utts) // num_cpus + 1
    utt_chunks = [utts[i:i+chunk_size] for i in range(0, len(utts), chunk_size)]
    
    st_time = time.time()
    # 创建进程池并行处理
    with multiprocessing.Pool(num_cpus) as pool:
        chunk_results  = pool.map(run_async_chunk, utt_chunks)
        
        all_results = []
        all_stats = []
        for chunk_result in chunk_results:
            for result, stat in chunk_result:
                if result:  # 过滤空结果
                    all_results.append(result)
                    all_stats.append(stat)

                
                
        
        print(f'final get {len(all_results)} result')
        with open(f'{output_dir}/all.jsonl', 'w') as f:
            for result in all_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        with open(f'{output_dir}/stats.jsonl', 'w') as f:
            for stat in all_stats:
                f.write(json.dumps(stat, ensure_ascii=False) + '\n')
                
    
    print(f'total use {time.time() - st_time} process {len(utts)} utts')
    analysis_time(all_stats)

if __name__ == "__main__":
    main()