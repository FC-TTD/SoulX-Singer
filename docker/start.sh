#!/usr/bin/env bash
set -euo pipefail

PRETRAINED_ROOT="${SOULX_PRETRAINED_ROOT:-/data/pretrained_models}"
PORT="${GRADIO_SERVER_PORT:-30931}"

mkdir -p "${PRETRAINED_ROOT}" /app/outputs "${HF_HOME:-/data/hf}"

python - <<'PY'
import os
from pathlib import Path

from huggingface_hub import snapshot_download

root = Path(os.environ.get("SOULX_PRETRAINED_ROOT", "/data/pretrained_models"))
singer_dir = root / "SoulX-Singer"
preprocess_dir = root / "SoulX-Singer-Preprocess"

required = [
    singer_dir / "model-svc.pt",
    singer_dir / "model.pt",
    preprocess_dir / "rmvpe" / "rmvpe.pt",
    preprocess_dir / "mel-band-roformer-karaoke" / "mel_band_roformer_karaoke_becruily.ckpt",
    preprocess_dir / "dereverb_mel_band_roformer" / "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
    preprocess_dir / "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    preprocess_dir / "parakeet-tdt-0.6b-v2" / "parakeet-tdt-0.6b-v2.nemo",
    preprocess_dir / "rosvot" / "rosvot" / "model.pt",
    preprocess_dir / "rosvot" / "rwbd" / "model.pt",
]

missing = [str(path) for path in required if not path.exists()]
print(f"missing_assets={missing}")

if missing:
    downloads = [
        (
            "Soul-AILab/SoulX-Singer",
            singer_dir,
            ["model.pt", "model-svc.pt"],
        ),
        (
            "Soul-AILab/SoulX-Singer-Preprocess",
            preprocess_dir,
            [
                "rmvpe/*",
                "mel-band-roformer-karaoke/*",
                "dereverb_mel_band_roformer/*",
                "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/*",
                "parakeet-tdt-0.6b-v2/*",
                "rosvot/*",
            ],
        ),
    ]

    for repo_id, local_dir, patterns in downloads:
        local_dir.mkdir(parents=True, exist_ok=True)
        print(f"snapshot_download repo={repo_id} allow_patterns={patterns}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            allow_patterns=patterns,
        )
PY

mkdir -p /app/pretrained_models
ln -sfn "${PRETRAINED_ROOT}/SoulX-Singer" /app/pretrained_models/SoulX-Singer
ln -sfn "${PRETRAINED_ROOT}/SoulX-Singer-Preprocess" /app/pretrained_models/SoulX-Singer-Preprocess

exec uvicorn api.app:app --host 0.0.0.0 --port "${PORT}"
