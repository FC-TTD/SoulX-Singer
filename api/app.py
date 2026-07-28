from contextlib import asynccontextmanager

from fastapi import FastAPI
import gradio as gr
import torch

import webui_combined
from api.routes import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(title="SoulX-Singer Fusion API", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }

app.include_router(router)

# Mount Gradio interface
gr_app = webui_combined.render_interface()
app = gr.mount_gradio_app(app, gr_app, path="")
