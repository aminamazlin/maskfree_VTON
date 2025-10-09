FROM python:3.10.16-slim-bookworm

# Set the working directory inside the container to /mlflow
WORKDIR /mlflow

# Copy the local requirements.txt from my computer into the container's current working directory
COPY requirements.txt .

# Docker runs this shell command during the build
# No cache: makes pip skip storing the download packages, which keeps the image smaller
RUN pip install --no-cache-dir -r requirements.txt