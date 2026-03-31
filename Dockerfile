FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN useradd -m -u 1000 appuser

WORKDIR /app

COPY requirements.txt .

# Step 1: install openenv-core WITH its deps so nothing gets missed
# (fastmcp, and any other undeclared-but-real imports all come in here)
RUN pip install --upgrade pip && \
    pip install openenv-core==0.2.3 || pip install --no-deps openenv-core==0.2.3

# Step 2: install the rest of requirements.txt.
# openai here will downgrade from whatever openenv-core tried to pull in
# (its metadata says >=2.7.2 which doesn't exist) to the real 1.x line.
# pip's --force-reinstall ensures our pin wins.
RUN pip install -r requirements.txt --force-reinstall

COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 7860

LABEL maintainer="DevOps AI Team"
LABEL version="5.0"
LABEL description="DevOps SRE AI Agent"

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]