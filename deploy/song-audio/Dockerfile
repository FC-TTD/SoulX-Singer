FROM soulx-native:h-344c8adf9d098c8b
COPY app.py worker.py /adapter/
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/adapter:/app HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
WORKDIR /app
ENTRYPOINT ["python", "-m", "uvicorn", "app:app", "--app-dir", "/adapter", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
