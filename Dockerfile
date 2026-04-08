FROM python:3.11-slim

WORKDIR /app

# Copy all files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run project
CMD ["python", "inference.py"]