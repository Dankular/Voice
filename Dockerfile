# Stage 1: fetch the browser VAD assets into public/vendor
FROM node:22-slim AS vendor
WORKDIR /w
COPY package.json package-lock.json ./
COPY scripts ./scripts
RUN npm ci --ignore-scripts && node scripts/vendor.mjs

# Stage 2: Python service (static page + TTS API)
FROM python:3.12-slim
WORKDIR /app
COPY requirements-tts.txt .
RUN pip install --no-cache-dir -r requirements-tts.txt
COPY app.py ./
COPY omnivoice_tts ./omnivoice_tts
COPY public ./public
COPY --from=vendor /w/public/vendor ./public/vendor
# Models (~430 MB) download on first TTS request into OMNIVOICE_HOME; point it at a persistent disk.
ENV OMNIVOICE_HOME=/data/omnivoice
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}"]
