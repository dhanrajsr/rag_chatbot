# Official Python image (Python 3.9)
FROM python:3.9-slim

# Working directory setup
WORKDIR /app

# System dependencies (if required)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt or direct installation
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copying remaining source code
COPY . .

# Port where FastAPI runs
EXPOSE 8000

# Running the application via Uvicorn server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
