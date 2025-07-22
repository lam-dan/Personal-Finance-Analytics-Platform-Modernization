FROM python:3.10-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the observability logging configuration from the correct project path
COPY observability/logging_config.py ./

CMD ["python", "logging_config.py"]