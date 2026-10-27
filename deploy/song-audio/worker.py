import json,sys,os
from pathlib import Path
os.chdir('/app');sys.path.insert(0,'/app')
import torch
assert torch.cuda.is_available(),'CUDA required; CPU/offload is not allowed'
import soundfile as sf
import librosa
import numpy as np
from webui_svc import AppState
p=Path(sys.argv[1]);j=json.loads(p.read_text());root=p.parent
# Existing native class/weights/FP16 path; clean stem inputs skip its separation stage.
state=AppState(use_fp16=True)
a,ma,pw,pf=state.run_preprocess(Path(j['reference']),root/'prompt',False)
b,mb,tw,tf=state.run_preprocess(Path(j['input']),root/'target',False)
assert a and b,(ma,mb)
assert np.mean(np.load(tf)>0)>=.01,'NO_VOICED_TARGET'
assert np.mean(np.load(pf)>0)>=.01,'NO_VOICED_REFERENCE'
ok,msg,result=state.run_svc(pw,tw,pf,tf,root,False,False,0,32,1.,42)
assert ok,msg
raw,sr=sf.read(result,always_2d=True,dtype='float32');duration=sf.info(tw).duration
if sr!=48000:raw=librosa.resample(raw.T,orig_sr=sr,target_sr=48000).T
frames=round(duration*48000);difference=len(raw)-frames
assert abs(difference)<=7200,'Generated duration differs by more than150ms; do not mix automatically'
fixed=raw[:frames] if len(raw)>=frames else np.pad(raw,((0,frames-len(raw)),(0,0)))
sf.write(root/'aligned.wav',fixed,48000,subtype='PCM_24')
j.update(status='completed',message='歌声转换完成',duration=frames/48000,sample_rate=48000,frames=frames,bytes=(root/'aligned.wav').stat().st_size,alignmentTrimFrames=max(0,difference),alignmentPadFrames=max(0,-difference),peakCudaBytes=torch.cuda.max_memory_allocated())
temp=p.with_suffix('.tmp');temp.write_text(json.dumps(j,ensure_ascii=False));temp.replace(p)
