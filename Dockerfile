# Stage 1: browser runtime assets (ONNX Runtime Web, Silero VAD) into public/vendor
FROM node:22-slim AS vendor
WORKDIR /w
COPY package.json package-lock.json ./
COPY scripts ./scripts
RUN npm ci --ignore-scripts && node scripts/vendor.mjs

# Stage 2: static files + voice-list proxy (no models run on the server)
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py get_voices.py ./
COPY public ./public
COPY --from=vendor /w/public/vendor ./public/vendor
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}"]
