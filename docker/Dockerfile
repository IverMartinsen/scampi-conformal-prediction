# Using lightweight Python image from Docker Hub as starting image
FROM --platform=linux/amd64 pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime
# Copy the current directory contents into the container at /app
COPY . /app
# Change the working directory
WORKDIR /app
# Install the required Python packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt