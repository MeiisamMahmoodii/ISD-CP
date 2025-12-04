# Use a lightweight Python base image (Multi-arch: works on x86 and ARM64/Mac)
FROM python:3.10-slim

# Set environment variables
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing pyc files
# PYTHONUNBUFFERED: Ensures console output is not buffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# git: useful for some pip packages
# curl/wget: useful for debugging
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app/

# Create a directory for checkpoints if it doesn't exist
RUN mkdir -p checkpoints_prod

# Expose port for TensorBoard (default 6006)
EXPOSE 6006

# Default command to run when the container starts
# We default to showing the help message
CMD ["python", "-m", "src.train.train", "--help"]
