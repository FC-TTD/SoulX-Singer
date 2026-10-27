"""Private adapter over the existing pinned SoulX native image, no model reimplementation."""
import contextvars
import asyncio,hashlib,json,math,os,re,subprocess,sys,time,uuid
from pathlib import Path
import numpy as np
import requests
import soundfile as sf
from fastapi import FastAPI,HTTPException,UploadFile,File,Form
from fastapi.responses import FileResponse
DATA=Path('/data');DATA.mkdir(exist_ok=True);app=FastAPI(title='Song SoulX SVC Preview');q=asyncio.Queue();busy=False
GPU=os.environ['AUDIO_GPU_UUID'];PN='http://192.168.100.18:17896/api/studio'
def write(path,data):
 path.parent.mkdir(parents=True,exist_ok=True);p=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp');p.write_text(json.dumps(data,ensure_ascii=False));p.replace(path)
def read(jid):
 if not re.fullmatch('[a-f0-9]{32}',jid) or not (DATA/jid/'job.json').exists():raise HTTPException(404,'任务不存在')
 return json.loads((DATA/jid/'job.json').read_text())
def public(j):return {k:v for k,v in j.items() if k not in ['input','reference'] and not k.startswith('_hub_')}
def verification_key():
 files=[Path('/app/pretrained_models/SoulX-Singer/model-svc.pt'),Path('/app/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt')]
 stats=[(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in files if p.is_file()]
 return hashlib.sha256(Path('/adapter/worker.py').read_bytes()+json.dumps(stats).encode()+b'344c8adf9d098c8b').hexdigest()
@app.get('/health')
def health():
 proof=json.loads((DATA/'verified.json').read_text()) if (DATA/'verified.json').is_file() else {}
 return {'status':'ok','busy':busy,'verified':proof.get('key')==verification_key(),'model':'SoulX-Singer SVC','nativeFp16':True}
@app.post('/jobs',status_code=202)
async def create(asset_id:str=Form(...),filename:str=Form('song.wav'),reference:UploadFile=File(...)):
 if runtime.status()['residency']=='unknown':raise HTTPException(503,'上次执行状态待核对，请稍后提交')
 if not re.fullmatch('[a-f0-9]{32}',asset_id):raise HTTPException(422,'输入音频无效')
 try:
  source=await asyncio.to_thread(lambda:requests.get(PN+'/jobs/'+asset_id,timeout=15).json());canonical=source['input_asset_id'];path=(Path('/studio-inputs')/canonical/'input.audio').resolve()
  if not re.fullmatch('[a-f0-9]{32}',canonical) or not path.is_relative_to('/studio-inputs') or not path.is_file():raise ValueError()
 except Exception:raise HTTPException(422,'中央音频不存在')
 data=await reference.read(32*1024**2+1)
 if not data or len(data)>32*1024**2:raise HTTPException(413,'参考歌声最多32 MB')
 jid=uuid.uuid4().hex;folder=DATA/jid;folder.mkdir();ref=folder/'reference.audio';ref.write_bytes(data)
 # Native audio loader validates codecs; ffprobe supplies a bounded reference duration.
 try:
  probe=await asyncio.create_subprocess_exec('ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',str(ref),stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE);out,_=await probe.communicate();duration=float(out.strip())
  if not 1<=duration<=30.1:raise ValueError()
 except Exception:raise HTTPException(422,'请提供1至30秒干净的参考歌声')
 j={'_hub_actor_id':os.environ.get('HUB_RUNTIME_ACTOR_ID'),'id':jid,'status':'queued','input':str(path),'reference':str(ref),'filename':filename,'createdAt':time.time(),'message':'排队等待显卡资源'};write(folder/'job.json',j);runtime.accept_background_work();await q.put((jid,contextvars.copy_context()));return public(j)
@app.get('/jobs/{jid}')
def state(jid:str):return public(read(jid))
@app.get('/jobs/{jid}/audio')
def audio(jid:str):
 j=read(jid)
 if j['status']!='completed':raise HTTPException(404,'输出尚未完成')
 return FileResponse(DATA/jid/'aligned.wav',media_type='audio/wav')
@app.get('/jobs/{jid}/audio/waveform')
def waveform(jid:str):
 j=read(jid)
 if j['status']!='completed':raise HTTPException(404,'输出尚未完成')
 p=DATA/jid/'peaks.json'
 if not p.exists():
  with sf.SoundFile(DATA/jid/'aligned.wav') as f:
   edges=np.linspace(0,len(f),901,dtype=int);peaks=[]
   for a,b in zip(edges[:-1],edges[1:]):f.seek(int(a));x=f.read(int(b-a),always_2d=True,dtype='float32');peaks.append(float(np.max(np.abs(x))) if len(x) else 0)
  write(p,{'peaks':peaks,'duration':j['duration']})
 return json.loads(p.read_text())
def run_native(jid):
 p=DATA/jid/'job.json'
 with runtime.execution():
  result=runtime.get().run([sys.executable,'/adapter/worker.py',str(p)],
   timeout=3600,log_path=str(DATA/jid/'native.log'))
  j=read(jid)
  if result['returncode'] or j['status']!='completed':
   log=(DATA/jid/'native.log').read_text()
   if 'NO_VOICED_' in log:raise RuntimeError('未检测到有效歌声音高，请选择有歌声的人声音轨与参考音频')
   raise RuntimeError('SoulX原生推理未完成，原分轨和参考歌声已保留')

async def worker():
 global busy
 while True:
  jid,context=await q.get();busy=True;p=DATA/jid/'job.json';j=read(jid);known_complete=True
  try:
   j.update(status='running',message='SoulX正在准备受管音高提取与音色转换');write(p,j)
   await asyncio.to_thread(context.run,run_native,jid)
   j=read(jid)
   write(DATA/'verified.json',{'image':'sha256:344c8adf9d098c8bd2d1b988df2b154b4951876e7218e1dc7f7aaad71ac8ce79','key':verification_key(),'jobId':jid,'native_fp16':True,'gpu':GPU,'at':time.time()})
  except Exception as exc:
   known_complete=getattr(exc,'code','') not in ['execution_unknown','node_unavailable']
   j=read(jid);j.update(status='failed' if known_complete else 'unknown',error=str(exc));write(p,j)
  finally:
   busy=False;q.task_done()
   if known_complete:context.run(runtime.complete_background_work)

@app.on_event('startup')
async def boot():
 for p in DATA.glob('*/job.json'):
  j=json.loads(p.read_text())
  if j.get('_hub_actor_id')==os.environ.get('HUB_RUNTIME_ACTOR_ID') and j['status'] in ['queued','running']:
   j.update(status='interrupted',error='服务重启，执行状态需要核对，未自动重跑');write(p,j)
 asyncio.create_task(worker())


def build_api(managed_runtime):
 global runtime
 runtime=managed_runtime
 runtime.pending_work=lambda:int(busy)+q.qsize()
 return app
