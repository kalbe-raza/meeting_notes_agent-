FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set defaults, but allow Render to override PORT
ENV HOST=0.0.0.0
ENV PORT=8000

# Use shell form so ${PORT} environment variable is expanded correctly by Render
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
