# Step 1: Base image with lightweight Python 3.10
FROM python:3.10-slim

# Step 2: Set working directory inside the container
WORKDIR /app

# Step 3: Install curl for container health checking
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy your FastAPI app and model files into the container
COPY app/ ./app/
COPY models/ ./models/

# Step 6: Expose port 8000 for network requests
EXPOSE 8000

# Step 7: Command to start the FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]