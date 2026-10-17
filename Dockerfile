FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so Docker can cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HuggingFace Spaces port
EXPOSE 7860

# Dashboard + API. For the plain log run use: docker run email-env python inference.py
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
