from abc import ABC, abstractmethod


from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


class VAD_engine_base(ABC):
    def __init__(self, params=None, device='cpu',):
        self.model_name = "base_model"
        self.model_space = "modelscope"
        self.model_vision = "base_vision"
        self.engine_name = '{}@{}@{}'.format(self.model_space, self.model_name, self.model_vision)
    
    @abstractmethod
    def vad_inference(self, speech_data, fs=16000):
        pass
    
    
    

class VAD_fsmn(VAD_engine_base):
    def __init__(self, params=None, device='cpu',):
        
        self.model_space = "modelscope"
        self.model_name = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
        self.model_vision = "v2.0.4"
        self.vad_model = pipeline(
                                task=Tasks.voice_activity_detection,
                                model=self.model_name,
                                model_revision=self.model_vision,
                                device=device,
                            )
        self.engine_name = '{}@{}@{}'.format(self.model_space, self.model_name, self.model_vision)
        
    def vad_inference(self, speech_data, fs=16000):
        vad_res = self.vad_model(input=speech_data)
        
        vad_anno = {}
        vad_anno['vad_space'] = self.model_space
        vad_anno['vad_model'] = self.model_name
        vad_anno['vad_model_vision'] = self.model_vision
        vad_anno['vad_segs'] = vad_res[0]['value']
        
        return vad_anno
    
    

if __name__ == '__main__':
    
    import soundfile as sf
    
    device = 'cuda:0'
    vad_model = VAD_fsmn(device=device)

    wav_path = '/data/nas/dataset/asr/kefu/shidian/wavs/date1211/cc-1866736084303028224.wav'
    
    speech_data, fs = sf.read(wav_path)
    
    res = vad_model.vad_inference(speech_data, fs=fs)
    print(res)