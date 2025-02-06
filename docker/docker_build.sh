#!/bin/bash

# Define the Docker image name
IMAGE_NAME="scampi"

# Build the Docker image
docker build -t $IMAGE_NAME .

# Print a message indicating the build is complete
echo "Docker image $IMAGE_NAME built successfully."

# tag the image
docker tag $IMAGE_NAME ivermartinsen/$IMAGE_NAME:latest

# push the image
docker push ivermartinsen/$IMAGE_NAME:latest