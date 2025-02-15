# Dockerfile

FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . .

# Expose any ports if a web service is added later
# EXPOSE 8000

# Default command: run the pipeline with example features
CMD ["python", "-m", "app.pipeline", "--features", "0.5", "-0.3"]
