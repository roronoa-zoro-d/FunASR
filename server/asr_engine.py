from abc import ABC, abstractmethod


from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


# 基类定义
class ASR_engine_base(ABC):
    def __init__(self, params=None, device='cpu',):
        self.model_name = "base_model"
        self.model_space = "modelscope"
        self.model_vision = "base_vision"
        self.engine_name = '{}@{}@{}'.format(self.model_space, self.model_name, self.model_vision)
        
        
    

    @abstractmethod
    def asr_inference(self, speech_data, segs = [],  fs=16000):
        pass
    
    

class ASR_ali_engine_base(ASR_engine_base):
    def __init__(self, params=None, device='cpu',):
        self.model_name = "base_model"
        self.model_space = "modelscope"
        self.model_vision = "base_vision"
        self.engine_name = '{}@{}@{}'.format(self.model_space, self.model_name, self.model_vision)
        
    
    @abstractmethod
    def load_model(self, params, device='cpu'):
        pass
    
    def asr_inference(self, speech_data, segs = [],  fs=16000):
        res = {}
        res['model_space'] = self.model_space
        res['model_name'] = self.model_name
        res['model_vision'] = self.model_vision
        
        res['segs'] = []
        
        if len(segs) == 0:
            asr_res = self.asr_model(speech_data, disable_pbar=True)
            text = asr_res[0]['text']
            res['full_text'] = text
            return res
        
        full_text =  ""
        for i, seg in enumerate(segs):
            st = int(seg[0]*fs/1000.0)
            ed = int(seg[1]*fs/1000.0)
            chunk = speech_data[st:ed]  
            asr_res = self.asr_model(input=chunk, disable_pbar=True)
            text = asr_res[0]['text']
            full_text += text
            res['segs'].append({"seg": seg, "text": text})
        
        res['full_text'] = full_text
        
        return res
    



class ASR_paraformer(ASR_ali_engine_base):
    def __init__(self, params=None, device='cpu',):
        super().__init__(params, device)
        self.load_model(params, device)
        
        
    
    def load_model(self, params, device='cpu'):
        self.model_name = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
        self.model_vision = "v2.0.4"
        self.engine_name = '{}@{}@{}'.format(self.model_space, self.model_name, self.model_vision)
        self.asr_model = pipeline(
                                task=Tasks.auto_speech_recognition,
                                model=self.model_name, 
                                model_revision=self.model_vision,
                                device=device,
                            )
        
        
        
        


class ASR_whisper_large(ASR_ali_engine_base):
    def __init__(self, params=None, device='cpu',):
        super().__init__(params, device)
        self.load_model(params, device)
        
        
    
    def load_model(self, params, device='cpu'):
        self.model_name = "iic/Whisper-large-v3"
        self.model_vision = "v2.0.5"
        self.engine_name = '{}@{}@{}'.format(self.model_space, self.model_name, self.model_vision)
        self.asr_model = pipeline(
                                task=Tasks.auto_speech_recognition,
                                model=self.model_name, 
                                model_revision=self.model_vision,
                                device=device,
                            )
        
        
        
class ASR_whisper_large_turbo(ASR_ali_engine_base):
    def __init__(self, params=None, device='cpu',):
        super().__init__(params, device)
        self.load_model(params, device)
        
        
    
    def load_model(self, params, device='cpu'):
        self.model_name = "iic/Whisper-large-v3-turbo"
        self.model_vision = "master"
        self.engine_name = '{}@{}@{}'.format(self.model_space, self.model_name, self.model_vision)
        self.asr_model = pipeline(
                                task=Tasks.auto_speech_recognition,
                                model=self.model_name, 
                                model_revision=self.model_vision,
                                device=device,
                            )
        
        
        


if __name__ == '__main__':
    import soundfile as sf
    
    vad_inference_pipeline = pipeline(
                                    task=Tasks.voice_activity_detection,
                                    model='iic/speech_fsmn_vad_zh-cn-16k-common-pytorch',
                                    model_revision="v2.0.4",
                                    device='cuda:1',
                                )
    
    
    wav_path = '/data/nas/dataset/asr/kefu/shidian/wavs/date1211/cc-1866736084303028224.wav'
    
    speech_data, fs = sf.read(wav_path)
    
    vad_res = vad_inference_pipeline(input=wav_path)
    segs = vad_res[0]['value']
    
    
    device = 'cuda:2'
    asr_paraformer = ASR_paraformer(device=device)
    asr_whisper_large = ASR_whisper_large(device=device)
    asr_whisper_large_turbo = ASR_whisper_large_turbo(device=device)
    
    
    res_paraformer = asr_paraformer.asr_inference(speech_data, segs, fs)
    print(f'paraformer: {res_paraformer}\n')
    
    res_whisper_large = asr_whisper_large.asr_inference(speech_data, segs, fs)
    print(f'whisper_large: {res_whisper_large}\n')
    
    res_whisper_large_turbo = asr_whisper_large_turbo.asr_inference(speech_data, segs, fs)
    print(f'whisper_large_turbo: {res_whisper_large_turbo}\n')
    
    
    print(f'-------------------------------')
    speech_data = speech_data[:int(fs*15)]
    res_paraformer = asr_paraformer.asr_inference(speech_data,  fs=fs)
    print(f'paraformer: {res_paraformer}\n')
    
    
    res_whisper_large = asr_whisper_large.asr_inference(speech_data,  fs=fs)
    print(f'whisper_large: {res_whisper_large}\n')
    
    
    res_whisper_large_turbo = asr_whisper_large_turbo.asr_inference(speech_data,  fs=fs)
    print(f'whisper_large_turbo: {res_whisper_large_turbo}\n')
    