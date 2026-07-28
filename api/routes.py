import traceback
from pathlib import Path

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
import shutil
import uuid

import webui
import webui_svc

router = APIRouter()

def apply_standard_postprocess(wav_path: Path):
    try:
        audio, sample_rate = sf.read(wav_path, always_2d=False)
        channels_first = audio.T if audio.ndim > 1 else audio
        channels_first, _ = librosa.effects.trim(channels_first, top_db=40)
        audio = channels_first.T if channels_first.ndim > 1 else channels_first

        meter = pyln.Meter(sample_rate)
        loudness = meter.integrated_loudness(audio)
        if np.isfinite(loudness):
            audio = pyln.normalize.loudness(audio, loudness, -16.0)

        sf.write(wav_path, audio, sample_rate)
    except Exception:
        traceback.print_exc()

@router.get("/languages")
async def get_languages():
    return {"languages": ["Mandarin", "Cantonese", "English"]}

@router.post("/api/tts")
async def api_tts(
    prompt_audio: UploadFile = File(...),
    target_audio: UploadFile = File(None),
    prompt_metadata: UploadFile = File(None),
    target_metadata: UploadFile = File(None),
    prompt_lyric_lang: str = Form("Mandarin"),
    target_lyric_lang: str = Form("Mandarin"),
    control: str = Form("score-controlled"),
    auto_shift: bool = Form(True),
    pitch_shift: int = Form(0),
    seed: int = Form(12306)
):
    app_state = webui.get_app_state()
    session_id = str(uuid.uuid4())[:8]
    session_base = Path(f"outputs/api/{session_id}")
    audio_dir = session_base / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    prompt_wav_path = audio_dir / "prompt.wav"
    with open(prompt_wav_path, "wb") as f:
        shutil.copyfileobj(prompt_audio.file, f)

    prompt_meta_path = session_base / "transcriptions" / "prompt" / "metadata.json"
    prompt_meta_path.parent.mkdir(parents=True, exist_ok=True)

    if prompt_metadata:
        with open(prompt_meta_path, "wb") as f:
            shutil.copyfileobj(prompt_metadata.file, f)
    else:
        # Generate metadata
        webui._trim_and_save_audio(str(prompt_wav_path), prompt_wav_path, webui.PROMPT_MAX_SEC_DEFAULT)
        ok, msg = app_state.run_preprocess(
            audio_path=prompt_wav_path,
            save_path=prompt_meta_path.parent,
            vocal_sep=False,
            lyric_lang=prompt_lyric_lang,
            max_merge_duration=webui.PROMPT_MAX_MERGE_DURATION_MS
        )
        if not ok:
            raise HTTPException(status_code=500, detail=f"Prompt preprocess failed: {msg}")

    target_meta_path = session_base / "transcriptions" / "target" / "metadata.json"
    target_meta_path.parent.mkdir(parents=True, exist_ok=True)

    if target_metadata:
        with open(target_meta_path, "wb") as f:
            shutil.copyfileobj(target_metadata.file, f)
    else:
        if not target_audio:
            raise HTTPException(status_code=400, detail="Either target_metadata or target_audio is required")
        target_wav_path = audio_dir / "target.wav"
        with open(target_wav_path, "wb") as f:
            shutil.copyfileobj(target_audio.file, f)

        webui._trim_and_save_audio(str(target_wav_path), target_wav_path, webui.TARGET_MAX_SEC_DEFAULT)
        ok, msg = app_state.run_preprocess(
            audio_path=target_wav_path,
            save_path=target_meta_path.parent,
            vocal_sep=True,
            lyric_lang=target_lyric_lang,
            max_merge_duration=webui.TARGET_MAX_MERGE_DURATION_MS
        )
        if not ok:
            raise HTTPException(status_code=500, detail=f"Target preprocess failed: {msg}")

    # SVS process
    ok, msg, merged = app_state.run_svs_from_paths(
        prompt_wav_path=str(prompt_wav_path),
        prompt_metadata_path=str(prompt_meta_path),
        target_metadata_path=str(target_meta_path),
        control=webui._control_to_internal(control),
        auto_shift=auto_shift,
        pitch_shift=pitch_shift,
        save_dir=session_base
    )
    if not ok or not merged:
        raise HTTPException(status_code=500, detail=f"SVS inference failed: {msg}")

    apply_standard_postprocess(merged)

    return FileResponse(merged, media_type="audio/wav")

@router.post("/api/svc")
async def api_svc(
    prompt_audio: UploadFile = File(...),
    target_audio: UploadFile = File(...),
    prompt_vocal_sep: bool = Form(False),
    target_vocal_sep: bool = Form(True),
    auto_shift: bool = Form(True),
    auto_mix_acc: bool = Form(True),
    pitch_shift: int = Form(0),
    n_step: int = Form(32),
    cfg: float = Form(1.0),
    seed: int = Form(42)
):
    app_state = webui_svc.get_app_state()
    session_id = str(uuid.uuid4())[:8]
    session_base = Path(f"outputs/api_svc/{session_id}")
    audio_dir = session_base / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    prompt_wav_path = audio_dir / "prompt.wav"
    with open(prompt_wav_path, "wb") as f:
        shutil.copyfileobj(prompt_audio.file, f)

    target_wav_path = audio_dir / "target.wav"
    with open(target_wav_path, "wb") as f:
        shutil.copyfileobj(target_audio.file, f)

    webui_svc._trim_and_save_audio(str(prompt_wav_path), prompt_wav_path, webui_svc.PROMPT_MAX_SEC_DEFAULT)
    webui_svc._trim_and_save_audio(str(target_wav_path), target_wav_path, webui_svc.TARGET_MAX_SEC_DEFAULT)

    prompt_ok, msg, prompt_wav, prompt_f0 = app_state.run_preprocess(
        audio_path=prompt_wav_path,
        save_path=session_base / "transcriptions" / "prompt",
        vocal_sep=prompt_vocal_sep
    )
    if not prompt_ok or not prompt_wav:
        raise HTTPException(status_code=500, detail=f"Prompt preprocess failed: {msg}")

    target_ok, msg, target_wav, target_f0 = app_state.run_preprocess(
        audio_path=target_wav_path,
        save_path=session_base / "transcriptions" / "target",
        vocal_sep=target_vocal_sep
    )
    if not target_ok or not target_wav:
        raise HTTPException(status_code=500, detail=f"Target preprocess failed: {msg}")

    ok, msg, generated = app_state.run_svc(
        prompt_wav_path=prompt_wav,
        target_wav_path=target_wav,
        prompt_f0_path=prompt_f0,
        target_f0_path=target_f0,
        session_base=session_base,
        auto_shift=auto_shift,
        auto_mix_acc=auto_mix_acc,
        pitch_shift=pitch_shift,
        n_step=n_step,
        cfg=cfg,
        seed=seed
    )

    if not ok or not generated:
        raise HTTPException(status_code=500, detail=f"SVC inference failed: {msg}")

    apply_standard_postprocess(generated)

    return FileResponse(generated, media_type="audio/wav")
